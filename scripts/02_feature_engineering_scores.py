"""
Phase 2 — Step 02 — Feature engineering: composite Usage / Trust / Frustration scores.

Each of the three headline scores is built from *multiple* survey columns
and rescaled to 0–100 so they are directly comparable.

  UsageScore       = mean of 4 normalised components, ×100
                     (frequency, agent depth, model breadth, workflow integration)

  TrustScore       = mean of 3 normalised components, ×100
                     (accuracy trust, complex-task trust, sentiment)

  FrustrationScore = mean of 2 normalised components, ×100
                     (problem count, threat perception)

We keep the per-component scores as columns too (UsageFreq, TrustAcc, …) so
the report can drill down to any single signal and so the ML script can
train on them as separate features.

Input  : output/cleaned_data/cleaned_survey_data   (Parquet from script 01)
Output : output/cleaned_data/ai_trust_scores       (Parquet, 49,191 rows)
"""
import os
from pathlib import Path

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

PROJECT_ROOT = Path("/users/sk7dn/big_data/AI_Trust_Paradox_Phase2")
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "output"
print("project root:", PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, size, split, array_except, array, lit, least
)

spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 02 Features (composite)")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(str(OUTPUT_DIR / "cleaned_data" / "cleaned_survey_data"))
print("loaded rows:", df.count())

# =============================================================================
# 2.1  USAGE  components  (4 columns → composite UsageScore on 0..100)
# =============================================================================
# 2.1a  Frequency (0..5) from AISelect
df = df.withColumn(
    "UsageFreq",
    when(col("AISelect") == "Yes, I use AI tools daily", 5)
    .when(col("AISelect") == "Yes, I use AI tools weekly", 4)
    .when(col("AISelect") == "Yes, I use AI tools monthly or infrequently", 3)
    .when(col("AISelect") == "No, but I plan to soon", 2)
    .when(col("AISelect") == "No, and I don't plan to", 1)
    .otherwise(0)  # Unknown / skipped
)

# 2.1b  Agent depth (0..5) from AIAgents
df = df.withColumn(
    "AgentDepth",
    when(col("AIAgents") == "Yes, I use AI agents at work daily", 5)
    .when(col("AIAgents") == "Yes, I use AI agents at work weekly", 4)
    .when(col("AIAgents") == "Yes, I use AI agents at work monthly or infrequently", 3)
    .when(col("AIAgents") == "No, I use AI exclusively in copilot/autocomplete mode", 2)
    .when(col("AIAgents") == "No, but I plan to", 1)
    .when(col("AIAgents") == "No, and I don't plan to", 0)
    .otherwise(0)
)

# 2.1c  Model breadth — # AI models worked with (capped at 10 to avoid outlier dominance)
df = df.withColumn(
    "AIModelCount",
    when(col("AIModelsHaveWorkedWith").isNull(), 0)
    .otherwise(size(split(col("AIModelsHaveWorkedWith"), ";")))
)
df = df.withColumn("AIModelCountCapped", least(col("AIModelCount"), lit(10)))

# 2.1d  Workflow integration — # of dev tasks done with AI
#       "Currently mostly AI"   counts full weight
#       "Currently partially AI" counts half weight
df = df.withColumn(
    "TasksMostlyAI",
    when(col("`AIToolCurrently mostly AI`").isNull(), 0)
    .otherwise(size(split(col("`AIToolCurrently mostly AI`"), ";")))
)
df = df.withColumn(
    "TasksPartiallyAI",
    when(col("`AIToolCurrently partially AI`").isNull(), 0)
    .otherwise(size(split(col("`AIToolCurrently partially AI`"), ";")))
)
df = df.withColumn(
    "WorkflowIntegration",
    least(col("TasksMostlyAI") + (col("TasksPartiallyAI") * lit(0.5)),
          lit(10.0))
)

# 2.1e  Composite UsageScore (0..100)
df = df.withColumn(
    "UsageScore",
    (
        (col("UsageFreq")           / 5.0) +
        (col("AgentDepth")          / 5.0) +
        (col("AIModelCountCapped")  / 10.0) +
        (col("WorkflowIntegration") / 10.0)
    ) / 4.0 * 100.0
)
print("\nUsageScore distribution:")
df.select("UsageFreq", "AgentDepth", "AIModelCount",
          "WorkflowIntegration", "UsageScore").describe().show()

# =============================================================================
# 2.2  TRUST  components  (3 columns → composite TrustScore on 0..100)
# =============================================================================
df = df.withColumn(
    "TrustAcc",
    when(col("AIAcc") == "Highly trust", 5)
    .when(col("AIAcc") == "Somewhat trust", 4)
    .when(col("AIAcc") == "Neither trust nor distrust", 3)
    .when(col("AIAcc") == "Somewhat distrust", 2)
    .when(col("AIAcc") == "Highly distrust", 1)
    .otherwise(None)
)
df = df.withColumn(
    "TrustComplex",
    when(col("AIComplex") == "Very well at handling complex tasks", 5)
    .when(col("AIComplex") == "Good, but not great at handling complex tasks", 4)
    .when(col("AIComplex") == "Neither good or bad at handling complex tasks", 3)
    .when(col("AIComplex") == "Bad at handling complex tasks", 2)
    .when(col("AIComplex") == "Very poor at handling complex tasks", 1)
    .otherwise(None)
)
df = df.withColumn(
    "Sentiment",
    when(col("AISent") == "Very favorable", 5)
    .when(col("AISent") == "Favorable", 4)
    .when(col("AISent") == "Indifferent", 3)
    .when(col("AISent") == "Unfavorable", 2)
    .when(col("AISent") == "Very unfavorable", 1)
    .otherwise(None)  # Unsure / NULL
)

# Composite TrustScore — requires *all three* components to be present.
# A developer who skipped any of these isn't included in the trust analysis.
df = df.withColumn(
    "TrustScore",
    when(col("TrustAcc").isNotNull() &
         col("TrustComplex").isNotNull() &
         col("Sentiment").isNotNull(),
        ( ((col("TrustAcc")     - 1) / 4.0) +
          ((col("TrustComplex") - 1) / 4.0) +
          ((col("Sentiment")    - 1) / 4.0) ) / 3.0 * 100.0
    ).otherwise(None)
)
print("\nTrustScore distribution (respondents who answered all 3 trust qs):")
df.filter(col("TrustScore").isNotNull()).select(
    "TrustAcc", "TrustComplex", "Sentiment", "TrustScore"
).describe().show()
print("rows with TrustScore:", df.filter(col("TrustScore").isNotNull()).count())

# =============================================================================
# 2.3  FRUSTRATION  components  (2 columns → composite FrustrationScore on 0..100)
# =============================================================================
NON_PROBLEM = [
    "I haven’t encountered any problems",  # U+2019 curly apostrophe
    "I don’t use AI tools regularly",
]
df = df.withColumn(
    "ProblemCount",
    when(col("AIFrustration").isNull(), 0)
    .when(col("AIFrustration") == "None reported", 0)
    .otherwise(
        size(array_except(
            split(col("AIFrustration"), ";"),
            array(*[lit(x) for x in NON_PROBLEM])
        ))
    )
)
df = df.withColumn("ProblemCountCapped", least(col("ProblemCount"), lit(5)))

df = df.withColumn(
    "ThreatLevel",
    when(col("AIThreat") == "Yes", 1.0)
    .when(col("AIThreat") == "I'm not sure", 0.5)
    .otherwise(0.0)  # No / NULL / Unknown
)

df = df.withColumn(
    "FrustrationScore",
    ( (col("ProblemCountCapped") / 5.0) + col("ThreatLevel") ) / 2.0 * 100.0
)
print("\nFrustrationScore distribution:")
df.select("ProblemCount", "ThreatLevel", "FrustrationScore").describe().show()

# =============================================================================
# 2.4  Auxiliary numeric features (used by the ML script)
# =============================================================================
df = df.withColumn(
    "DevEnvToolCount",
    when(col("DevEnvsHaveWorkedWith").isNull(), 0)
    .otherwise(size(split(col("DevEnvsHaveWorkedWith"), ";")))
)

# =============================================================================
# 2.5  Persist the feature-engineered dataset
# =============================================================================
out = OUTPUT_DIR / "cleaned_data" / "ai_trust_scores"
df.write.mode("overwrite").parquet(str(out))
print("\nwrote:", out)

print("\n--- sample rows (selected score columns) ---")
df.select(
    "ResponseId",
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    "UsageScore",
    "TrustAcc", "TrustComplex", "Sentiment",
    "TrustScore",
    "ProblemCount", "ThreatLevel",
    "FrustrationScore",
).show(5, truncate=False)

spark.stop()
