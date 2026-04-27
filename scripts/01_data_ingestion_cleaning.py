#!/usr/bin/env python
# coding: utf-8

# # Notebook 01 — Data Ingestion and Cleaning
# 
# Loads the 2025 Stack Overflow Developer Survey, selects the columns we need for the AI-trust analysis, normalises missing values, casts experience fields to numeric, and writes the cleaned data as Parquet.
# 
# **Input:** `data/survey_results_public.csv` (49,191 rows × 172 cols)
# 
# **Output:** `output/cleaned_data/cleaned_survey_data` (Parquet)

# In[1]:


import os, sys

from runtime_env import configure_spark_runtime, project_root

PROJECT_ROOT = project_root(__file__)
configure_spark_runtime(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
print("project root:", PROJECT_ROOT)


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 01 Ingestion")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("Spark version:", spark.version)


# ## 1.1  Load the raw CSV
# 
# The survey CSV uses the literal string `"NA"` for missing answers, so we register that as the null sentinel up front. Many free-text fields contain embedded newlines and quoted commas — we enable `multiLine` and the embedded-quote escape so Spark parses the file correctly.

# In[3]:


df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .option("nullValue", "NA")
    .csv(str(DATA_DIR / "survey_results_public.csv"))
)
print("Rows :", df.count())
print("Cols :", len(df.columns))


# In[4]:


df.printSchema()


# ## 1.2  Select the columns we care about
# 
# Out of 172 columns we keep ~30: developer profile (role, experience, country) plus all the AI usage / trust / frustration / agent fields we need for the three composite scores.

# In[5]:


selected_columns = [
    # ---- developer profile ----
    "ResponseId", "MainBranch", "Age", "EdLevel", "Employment",
    "WorkExp", "YearsCode", "DevType", "OrgSize", "ICorPM",
    "RemoteWork", "Country", "Industry", "ConvertedCompYearly", "JobSat",
    # ---- AI sentiment / trust ----
    "AIThreat", "SOFriction",
    "AISelect", "AISent", "AIAcc", "AIComplex",
    "AIToolCurrently partially AI", "AIToolCurrently mostly AI",
    "AIToolPlan to partially use AI", "AIToolPlan to mostly use AI",
    "AIToolDon't plan to use AI for this task",
    "AIFrustration", "AIExplain",
    # ---- AI agents ----
    "AIAgents", "AIAgentChange", "AIAgent_Uses", "AgentUsesGeneral",
    # ---- AI tooling ----
    "AIModelsHaveWorkedWith", "DevEnvsHaveWorkedWith",
]
available = [c for c in selected_columns if c in df.columns]
missing   = [c for c in selected_columns if c not in df.columns]
print(f"available: {len(available)} cols")
print(f"missing  : {missing}")
df_selected = df.select(available)
df_selected.show(3, truncate=60, vertical=True)


# ## 1.3  De-duplicate and normalise nulls
# 
# `ResponseId` should be unique already, but we drop dupes defensively. Then we replace nulls in categorical fields with `"Unknown"` so they still appear (instead of being dropped) in `GROUP BY` queries downstream.

# In[6]:


df_clean = df_selected.dropDuplicates(["ResponseId"])
df_clean = df_clean.fillna({
    "AISelect"      : "Unknown",
    "AISent"        : "Unknown",
    "AIAcc"         : "Unknown",
    "AIComplex"     : "Unknown",
    "AIFrustration" : "None reported",
    "AIThreat"      : "Unknown",
    "SOFriction"    : "Unknown",
    "AIAgents"      : "Unknown",
    "DevType"       : "Unknown",
    "Country"       : "Unknown",
    "Industry"      : "Unknown",
    "RemoteWork"    : "Unknown",
})
print("rows after de-dupe:", df_clean.count())


# ## 1.4  Convert experience fields to numeric
# 
# `WorkExp` is just years-as-a-string; cast it to double. `YearsCode` has two non-numeric edge values that we map by hand.

# In[7]:


df_clean = df_clean.withColumn("WorkExpNum", col("WorkExp").cast("double"))
df_clean = df_clean.withColumn(
    "YearsCodeNum",
    when(col("YearsCode") == "Less than 1 year", 0.0)
    .when(col("YearsCode") == "More than 50 years", 51.0)
    .otherwise(col("YearsCode").cast("double"))
)
df_clean.select("WorkExp", "WorkExpNum", "YearsCode", "YearsCodeNum").show(5)


# ## 1.5  Persist the cleaned dataset as Parquet
# 
# Parquet is columnar + compressed, so the ~30 columns we kept fit in a few MB and load orders-of-magnitude faster than re-parsing the CSV.

# In[8]:


out = OUTPUT_DIR / "cleaned_data" / "cleaned_survey_data"
df_clean.write.mode("overwrite").parquet(str(out))
print("wrote:", out)
print("partitions written:", len(list(out.glob("part-*.parquet"))))


# In[9]:


spark.stop()
