#!/usr/bin/env python
# coding: utf-8

# # Notebook 02 — Feature Engineering
# 
# Builds the numeric scores that the rest of the analysis pivots on:
# - **UsageScore** (1–5) from `AISelect`
# - **TrustScore** (1–5) from `AIAcc`
# - **SentimentScore** (1–5) from `AISent`
# - **ComplexityScore** (1–5) from `AIComplex`
# - **FrustrationScore** (count of problem tokens) from `AIFrustration`
# - **AgentAdoptionScore** (0–5) from `AIAgents`
# - **AIModelCount** (cardinality of `AIModelsHaveWorkedWith`)
# - **DevEnvToolCount** (cardinality of `DevEnvsHaveWorkedWith`)
# - Two composites — `OverallTrustScore` and `OverallUsageScore`

# In[1]:


import os, sys
from pathlib import Path

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

PROJECT_ROOT = Path("/users/sk7dn/big_data/AI_Trust_Paradox_Phase2")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
print("project root:", PROJECT_ROOT)


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, size, split, array_except, array, lit
)

spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 02 Features")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(str(OUTPUT_DIR / "cleaned_data" / "cleaned_survey_data"))
print("loaded rows:", df.count())


# ## 2.1  Usage Score (1–5) from `AISelect`
# 
# Higher score = more frequent AI usage. `Unknown` (people who skipped) gets 0 so they're easy to filter out later.

# In[3]:


df_scores = df.withColumn(
    "UsageScore",
    when(col("AISelect") == "Yes, I use AI tools daily", 5)
    .when(col("AISelect") == "Yes, I use AI tools weekly", 4)
    .when(col("AISelect") == "Yes, I use AI tools monthly or infrequently", 3)
    .when(col("AISelect") == "No, but I plan to soon", 2)
    .when(col("AISelect") == "No, and I don't plan to", 1)
    .otherwise(0)
)
df_scores.groupBy("AISelect", "UsageScore").count().orderBy("UsageScore").show(truncate=False)


# ## 2.2  Trust Score (1–5) from `AIAcc`
# 
# `AIAcc` directly asks how much developers trust the accuracy of AI output.

# In[4]:


df_scores = df_scores.withColumn(
    "TrustScore",
    when(col("AIAcc") == "Highly trust", 5)
    .when(col("AIAcc") == "Somewhat trust", 4)
    .when(col("AIAcc") == "Neither trust nor distrust", 3)
    .when(col("AIAcc") == "Somewhat distrust", 2)
    .when(col("AIAcc") == "Highly distrust", 1)
    .otherwise(None)
)
df_scores.groupBy("AIAcc", "TrustScore").count().orderBy("TrustScore").show(truncate=False)


# ## 2.3  Sentiment Score (1–5) from `AISent`
# 
# Favorability toward AI tools as a category. `Unsure` is dropped (None).

# In[5]:


df_scores = df_scores.withColumn(
    "SentimentScore",
    when(col("AISent") == "Very favorable", 5)
    .when(col("AISent") == "Favorable", 4)
    .when(col("AISent") == "Indifferent", 3)
    .when(col("AISent") == "Unfavorable", 2)
    .when(col("AISent") == "Very unfavorable", 1)
    .otherwise(None)
)


# ## 2.4  Complexity Score (1–5) from `AIComplex`
# 
# How well AI tools handle complex tasks. The opt-out option *"I don't use AI tools for complex tasks / I don't know"* maps to None.

# In[6]:


df_scores = df_scores.withColumn(
    "ComplexityScore",
    when(col("AIComplex") == "Very well at handling complex tasks", 5)
    .when(col("AIComplex") == "Good, but not great at handling complex tasks", 4)
    .when(col("AIComplex") == "Neither good or bad at handling complex tasks", 3)
    .when(col("AIComplex") == "Bad at handling complex tasks", 2)
    .when(col("AIComplex") == "Very poor at handling complex tasks", 1)
    .otherwise(None)
)


# ## 2.5  Frustration Score from `AIFrustration` (multi-select)
# 
# `AIFrustration` is a `;`-delimited multi-select. We count how many *problem* tokens were chosen, after subtracting the two non-problem options ("haven't encountered any problems" and "don't use AI tools regularly"). Note the survey uses curly apostrophes (`U+2019`).

# In[7]:


NON_PROBLEM = [
    "I haven’t encountered any problems",
    "I don’t use AI tools regularly",
]
df_scores = df_scores.withColumn(
    "FrustrationScore",
    when(col("AIFrustration").isNull(), 0)
    .when(col("AIFrustration") == "None reported", 0)
    .otherwise(
        size(array_except(
            split(col("AIFrustration"), ";"),
            array(*[lit(x) for x in NON_PROBLEM])
        ))
    )
)
df_scores.groupBy("FrustrationScore").count().orderBy("FrustrationScore").show()


# ## 2.6  Agent Adoption Score (0–5) from `AIAgents`

# In[8]:


df_scores = df_scores.withColumn(
    "AgentAdoptionScore",
    when(col("AIAgents") == "Yes, I use AI agents at work daily", 5)
    .when(col("AIAgents") == "Yes, I use AI agents at work weekly", 4)
    .when(col("AIAgents") == "Yes, I use AI agents at work monthly or infrequently", 3)
    .when(col("AIAgents") == "No, I use AI exclusively in copilot/autocomplete mode", 2)
    .when(col("AIAgents") == "No, but I plan to", 1)
    .when(col("AIAgents") == "No, and I don't plan to", 0)
    .otherwise(0)
)


# ## 2.7  Cardinality features — # AI models & # IDEs used

# In[9]:


df_scores = df_scores.withColumn(
    "AIModelCount",
    when(col("AIModelsHaveWorkedWith").isNull(), 0)
    .otherwise(size(split(col("AIModelsHaveWorkedWith"), ";")))
)
df_scores = df_scores.withColumn(
    "DevEnvToolCount",
    when(col("DevEnvsHaveWorkedWith").isNull(), 0)
    .otherwise(size(split(col("DevEnvsHaveWorkedWith"), ";")))
)
df_scores.select("AIModelCount", "DevEnvToolCount").describe().show()


# ## 2.8  Composite scores

# In[10]:


df_scores = df_scores.withColumn(
    "OverallTrustScore",
    (col("TrustScore") + col("SentimentScore") + col("ComplexityScore")) / 3
)
df_scores = df_scores.withColumn(
    "OverallUsageScore",
    col("UsageScore") + col("AgentAdoptionScore") + col("AIModelCount")
)
df_scores.select(
    "UsageScore", "TrustScore", "SentimentScore", "ComplexityScore",
    "FrustrationScore", "AgentAdoptionScore", "AIModelCount",
    "DevEnvToolCount", "OverallTrustScore", "OverallUsageScore"
).describe().show()


# ## 2.9  Persist feature-engineered Parquet

# In[11]:


out = OUTPUT_DIR / "cleaned_data" / "ai_trust_scores"
df_scores.write.mode("overwrite").parquet(str(out))
print("wrote:", out)


# In[12]:


spark.stop()

