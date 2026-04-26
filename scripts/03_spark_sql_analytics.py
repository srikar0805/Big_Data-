#!/usr/bin/env python
# coding: utf-8

# # Notebook 03 — Spark SQL Analytics
# 
# Runs the seven analytical queries that drive the report:
# 1. Dataset summary
# 2. AI usage distribution
# 3. Trust by AI usage
# 4. Trust by developer role
# 5. Trust by experience level
# 6. **AI Trust Paradox quadrant analysis** — the core finding
# 7. Paradox group profile (role × country × remote)

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
from pyspark.sql.functions import avg, col, when

spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 03 SparkSQL")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df_scores = spark.read.parquet(str(OUTPUT_DIR / "cleaned_data" / "ai_trust_scores"))
df_scores.createOrReplaceTempView("ai_trust")
print("rows:", df_scores.count())


# ## Query 1 — Dataset summary

# In[3]:


summary = spark.sql("""
SELECT
    COUNT(*)                           AS TotalResponses,
    COUNT(DISTINCT Country)            AS CountryCount,
    COUNT(DISTINCT DevType)            AS DeveloperRoleCount,
    AVG(UsageScore)                    AS AvgUsageScore,
    AVG(TrustScore)                    AS AvgTrustScore,
    AVG(FrustrationScore)              AS AvgFrustrationScore
FROM ai_trust
""")
summary.show(truncate=False)
summary.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "dataset_summary"), header=True)


# ## Query 2 — AI usage distribution

# In[4]:


ai_usage_distribution = spark.sql("""
SELECT AISelect, COUNT(*) AS DeveloperCount
FROM ai_trust
GROUP BY AISelect
ORDER BY DeveloperCount DESC
""")
ai_usage_distribution.show(truncate=False)
ai_usage_distribution.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "ai_usage_distribution"), header=True)


# ## Query 3 — Trust by AI usage

# In[5]:


trust_by_usage = spark.sql("""
SELECT
    AISelect,
    COUNT(*)                AS DeveloperCount,
    AVG(TrustScore)         AS AvgTrustScore,
    AVG(SentimentScore)     AS AvgSentimentScore,
    AVG(ComplexityScore)    AS AvgComplexityScore,
    AVG(FrustrationScore)   AS AvgFrustrationScore
FROM ai_trust
GROUP BY AISelect
ORDER BY AvgTrustScore DESC
""")
trust_by_usage.show(truncate=False)
trust_by_usage.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "trust_by_usage"), header=True)


# ## Query 4 — Trust by developer role

# In[6]:


trust_by_role = spark.sql("""
SELECT
    DevType,
    COUNT(*)                AS DeveloperCount,
    AVG(UsageScore)         AS AvgUsageScore,
    AVG(TrustScore)         AS AvgTrustScore,
    AVG(FrustrationScore)   AS AvgFrustrationScore
FROM ai_trust
WHERE DevType IS NOT NULL AND DevType <> 'Unknown'
GROUP BY DevType
HAVING DeveloperCount >= 100
ORDER BY AvgTrustScore DESC
""")
trust_by_role.show(30, truncate=False)
trust_by_role.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "trust_by_role"), header=True)


# ## Query 5 — Trust by experience level

# In[7]:


experience_analysis = spark.sql("""
SELECT
    CASE
        WHEN WorkExpNum IS NULL                   THEN 'Unknown'
        WHEN WorkExpNum < 2                       THEN 'Beginner: 0-1 years'
        WHEN WorkExpNum BETWEEN 2  AND 5          THEN 'Junior: 2-5 years'
        WHEN WorkExpNum BETWEEN 6  AND 10         THEN 'Mid-level: 6-10 years'
        WHEN WorkExpNum BETWEEN 11 AND 20         THEN 'Senior: 11-20 years'
        ELSE 'Expert: 20+ years'
    END                                              AS ExperienceGroup,
    COUNT(*)                AS DeveloperCount,
    AVG(UsageScore)         AS AvgUsageScore,
    AVG(TrustScore)         AS AvgTrustScore,
    AVG(FrustrationScore)   AS AvgFrustrationScore
FROM ai_trust
GROUP BY 1
ORDER BY AvgTrustScore DESC
""")
experience_analysis.show(truncate=False)
experience_analysis.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "experience_analysis"), header=True)


# ## Query 6 — AI Trust Paradox quadrant analysis ⭐
# 
# Split the population into four groups around the **mean** UsageScore and TrustScore. **High Usage + Low Trust** is the *paradox group*.

# In[8]:


avg_usage = df_scores.agg(avg("UsageScore")).collect()[0][0]
avg_trust = df_scores.agg(avg("TrustScore")).collect()[0][0]
print(f"avg UsageScore: {avg_usage:.3f}")
print(f"avg TrustScore: {avg_trust:.3f}")

df_quadrants = df_scores.withColumn(
    "TrustUsageGroup",
    when((col("UsageScore") >= avg_usage) & (col("TrustScore") >= avg_trust),
         "High Usage - High Trust")
    .when((col("UsageScore") >= avg_usage) & (col("TrustScore") <  avg_trust),
         "High Usage - Low Trust")
    .when((col("UsageScore") <  avg_usage) & (col("TrustScore") >= avg_trust),
         "Low Usage - High Trust")
    .otherwise("Low Usage - Low Trust")
)
df_quadrants.createOrReplaceTempView("ai_trust_quadrants")

quadrant_summary = spark.sql("""
SELECT
    TrustUsageGroup,
    COUNT(*)                  AS DeveloperCount,
    AVG(UsageScore)           AS AvgUsageScore,
    AVG(TrustScore)           AS AvgTrustScore,
    AVG(FrustrationScore)     AS AvgFrustrationScore,
    AVG(AgentAdoptionScore)   AS AvgAgentAdoptionScore
FROM ai_trust_quadrants
GROUP BY TrustUsageGroup
ORDER BY DeveloperCount DESC
""")
quadrant_summary.show(truncate=False)
quadrant_summary.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "quadrant_summary"), header=True)


# ## Query 7 — Profile of the paradox group

# In[9]:


paradox_profile = spark.sql("""
SELECT
    DevType,
    Country,
    RemoteWork,
    COUNT(*)                  AS DeveloperCount,
    AVG(WorkExpNum)           AS AvgWorkExperience,
    AVG(FrustrationScore)     AS AvgFrustration,
    AVG(AgentAdoptionScore)   AS AvgAgentAdoption
FROM ai_trust_quadrants
WHERE TrustUsageGroup = 'High Usage - Low Trust'
  AND DevType IS NOT NULL AND DevType <> 'Unknown'
GROUP BY DevType, Country, RemoteWork
HAVING DeveloperCount >= 20
ORDER BY DeveloperCount DESC
""")
paradox_profile.show(30, truncate=False)
paradox_profile.coalesce(1).write.mode("overwrite").csv(
    str(OUTPUT_DIR / "spark_sql_results" / "paradox_profile"), header=True)


# ## Persist quadrants for downstream notebooks

# In[10]:


out = OUTPUT_DIR / "cleaned_data" / "ai_trust_quadrants"
df_quadrants.write.mode("overwrite").parquet(str(out))
print("wrote:", out)
spark.stop()

