"""
Phase 2 — Step 03 — Spark-SQL analytics on the composite scores.

Runs the seven analytical queries that drive the report:
  1. Dataset summary
  2. AI-usage-frequency distribution
  3. Trust by AI-usage band  (low/med/high)
  4. Trust by developer role
  5. Trust by experience level
  6. AI Trust Paradox quadrant analysis (split on the **median** of each composite)
  7. Profile of the paradox group (role × country × remote)

Reads  : output/cleaned_data/ai_trust_scores
Writes : output/cleaned_data/ai_trust_quadrants  (parquet)
         output/spark_sql_results/*              (one CSV per query)
"""
from runtime_env import configure_spark_runtime, project_root

PROJECT_ROOT = project_root(__file__)
configure_spark_runtime(PROJECT_ROOT)
OUTPUT_DIR   = PROJECT_ROOT / "output"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

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

def save(df, name):
    df.coalesce(1).write.mode("overwrite").csv(
        str(OUTPUT_DIR / "spark_sql_results" / name), header=True)

# -----------------------------------------------------------------------------
# Query 1 — dataset summary
# -----------------------------------------------------------------------------
print("\n=== Q1: dataset summary ===")
summary = spark.sql("""
SELECT
    COUNT(*)                           AS TotalResponses,
    COUNT(TrustScore)                  AS RespondentsWithTrust,
    COUNT(DISTINCT Country)            AS CountryCount,
    COUNT(DISTINCT DevType)            AS DeveloperRoleCount,
    AVG(UsageScore)                    AS AvgUsageScore,
    AVG(TrustScore)                    AS AvgTrustScore,
    AVG(FrustrationScore)              AS AvgFrustrationScore
FROM ai_trust
""")
summary.show(truncate=False)
save(summary, "dataset_summary")

# -----------------------------------------------------------------------------
# Query 2 — AI usage distribution (raw frequency category)
# -----------------------------------------------------------------------------
print("\n=== Q2: AI usage distribution ===")
ai_usage_distribution = spark.sql("""
SELECT AISelect, COUNT(*) AS DeveloperCount
FROM ai_trust
GROUP BY AISelect
ORDER BY DeveloperCount DESC
""")
ai_usage_distribution.show(truncate=False)
save(ai_usage_distribution, "ai_usage_distribution")

# -----------------------------------------------------------------------------
# Query 3 — Trust by AI-usage band (composite)
# -----------------------------------------------------------------------------
print("\n=== Q3: Trust by usage band ===")
trust_by_usage = spark.sql("""
SELECT
    CASE
      WHEN UsageScore < 25 THEN 'Low (0-25)'
      WHEN UsageScore < 50 THEN 'Medium-Low (25-50)'
      WHEN UsageScore < 75 THEN 'Medium-High (50-75)'
      ELSE                       'High (75-100)'
    END                              AS UsageBand,
    COUNT(*)                         AS DeveloperCount,
    AVG(UsageScore)                  AS AvgUsageScore,
    AVG(TrustScore)                  AS AvgTrustScore,
    AVG(FrustrationScore)            AS AvgFrustrationScore
FROM ai_trust
GROUP BY 1
ORDER BY AvgUsageScore
""")
trust_by_usage.show(truncate=False)
save(trust_by_usage, "trust_by_usage")

# -----------------------------------------------------------------------------
# Query 4 — Trust by developer role
# -----------------------------------------------------------------------------
print("\n=== Q4: Trust by developer role ===")
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
save(trust_by_role, "trust_by_role")

# -----------------------------------------------------------------------------
# Query 5 — Trust by experience level
# -----------------------------------------------------------------------------
print("\n=== Q5: Trust by experience level ===")
experience_analysis = spark.sql("""
SELECT
    CASE
        WHEN WorkExpNum IS NULL              THEN 'Unknown'
        WHEN WorkExpNum < 2                  THEN 'Beginner: 0-1 years'
        WHEN WorkExpNum BETWEEN 2  AND 5     THEN 'Junior: 2-5 years'
        WHEN WorkExpNum BETWEEN 6  AND 10    THEN 'Mid-level: 6-10 years'
        WHEN WorkExpNum BETWEEN 11 AND 20    THEN 'Senior: 11-20 years'
        ELSE 'Expert: 20+ years'
    END                                                AS ExperienceGroup,
    COUNT(*)                AS DeveloperCount,
    AVG(UsageScore)         AS AvgUsageScore,
    AVG(TrustScore)         AS AvgTrustScore,
    AVG(FrustrationScore)   AS AvgFrustrationScore
FROM ai_trust
GROUP BY 1
ORDER BY AvgTrustScore DESC
""")
experience_analysis.show(truncate=False)
save(experience_analysis, "experience_analysis")

# -----------------------------------------------------------------------------
# Query 6 — AI Trust Paradox quadrant analysis  (median split)
# -----------------------------------------------------------------------------
print("\n=== Q6: Trust Paradox quadrant analysis ===")
analysed = df_scores.filter(col("TrustScore").isNotNull())
median_usage = analysed.approxQuantile("UsageScore", [0.5], 0.001)[0]
median_trust = analysed.approxQuantile("TrustScore", [0.5], 0.001)[0]
print(f"median UsageScore (analysed population): {median_usage:.2f}")
print(f"median TrustScore (analysed population): {median_trust:.2f}")

df_quadrants = analysed.withColumn(
    "TrustUsageGroup",
    when((col("UsageScore") >= median_usage) & (col("TrustScore") >= median_trust),
         "High Usage - High Trust")
    .when((col("UsageScore") >= median_usage) & (col("TrustScore") <  median_trust),
         "High Usage - Low Trust")
    .when((col("UsageScore") <  median_usage) & (col("TrustScore") >= median_trust),
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
    AVG(AgentDepth)           AS AvgAgentDepth,
    AVG(AIModelCount)         AS AvgModelCount,
    AVG(WorkflowIntegration)  AS AvgWorkflowIntegration
FROM ai_trust_quadrants
GROUP BY TrustUsageGroup
ORDER BY DeveloperCount DESC
""")
quadrant_summary.show(truncate=False)
save(quadrant_summary, "quadrant_summary")

# -----------------------------------------------------------------------------
# Query 7 — Profile of the paradox group
# -----------------------------------------------------------------------------
print("\n=== Q7: Paradox group profile ===")
paradox_profile = spark.sql("""
SELECT
    DevType,
    Country,
    RemoteWork,
    COUNT(*)                  AS DeveloperCount,
    AVG(WorkExpNum)           AS AvgWorkExperience,
    AVG(FrustrationScore)     AS AvgFrustration,
    AVG(AgentDepth)           AS AvgAgentDepth
FROM ai_trust_quadrants
WHERE TrustUsageGroup = 'High Usage - Low Trust'
  AND DevType IS NOT NULL AND DevType <> 'Unknown'
GROUP BY DevType, Country, RemoteWork
HAVING DeveloperCount >= 20
ORDER BY DeveloperCount DESC
""")
paradox_profile.show(30, truncate=False)
save(paradox_profile, "paradox_profile")

# -----------------------------------------------------------------------------
# Persist quadrants for downstream scripts
# -----------------------------------------------------------------------------
out = OUTPUT_DIR / "cleaned_data" / "ai_trust_quadrants"
df_quadrants.write.mode("overwrite").parquet(str(out))
print("\nwrote:", out)

spark.stop()
