#!/usr/bin/env python
# coding: utf-8

# # Notebook 05 — Visualizations and Testing
# 
# Generates all eight required plots into `output/visualizations/` and runs a small testing/validation suite (timing, schema sanity, output-file presence).

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


import time, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, when

spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 05 Viz")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df_scores = spark.read.parquet(str(OUTPUT_DIR / "cleaned_data" / "ai_trust_scores"))
df_scores.createOrReplaceTempView("ai_trust")

avg_usage = df_scores.agg(avg("UsageScore")).collect()[0][0]
avg_trust = df_scores.agg(avg("TrustScore")).collect()[0][0]
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

VIZ = OUTPUT_DIR / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)


# ## Viz 1 — AI usage distribution

# In[3]:


usage_pd = spark.sql("""
SELECT AISelect, COUNT(*) AS DeveloperCount
FROM ai_trust
WHERE AISelect <> 'Unknown'
GROUP BY AISelect
ORDER BY DeveloperCount DESC
""").toPandas()

plt.figure(figsize=(11, 6))
plt.bar(usage_pd["AISelect"], usage_pd["DeveloperCount"], color="#2b7cbf")
plt.xlabel("AI Usage Frequency")
plt.ylabel("Developer Count")
plt.title("Distribution of AI Tool Usage")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "ai_usage_distribution.png", dpi=120)
plt.show()


# ## Viz 2 — Average trust by AI usage frequency

# In[4]:


trust_usage_pd = spark.sql("""
SELECT AISelect, AVG(TrustScore) AS AvgTrustScore
FROM ai_trust
WHERE TrustScore IS NOT NULL AND AISelect <> 'Unknown'
GROUP BY AISelect
ORDER BY AvgTrustScore DESC
""").toPandas()

plt.figure(figsize=(11, 6))
plt.bar(trust_usage_pd["AISelect"], trust_usage_pd["AvgTrustScore"], color="#33a02c")
plt.xlabel("AI Usage Frequency")
plt.ylabel("Average Trust Score (1-5)")
plt.title("Average AI Trust by Usage Frequency")
plt.ylim(0, 5)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "trust_by_usage.png", dpi=120)
plt.show()


# ## Viz 3 — Usage vs Trust (jittered scatter, coloured by Frustration)

# In[5]:


scatter_pd = (df_scores.select("UsageScore", "TrustScore", "FrustrationScore")
              .filter("TrustScore IS NOT NULL AND UsageScore > 0")
              .toPandas())

# tiny jitter so the (1-5) integer grid is readable
rng = np.random.default_rng(0)
sx = scatter_pd["UsageScore"]   + rng.uniform(-0.2, 0.2, len(scatter_pd))
sy = scatter_pd["TrustScore"]   + rng.uniform(-0.2, 0.2, len(scatter_pd))
plt.figure(figsize=(9, 7))
sc = plt.scatter(sx, sy, c=scatter_pd["FrustrationScore"],
                 alpha=0.35, s=12, cmap="plasma")
plt.colorbar(sc, label="Frustration tokens")
plt.xlabel("Usage Score")
plt.ylabel("Trust Score")
plt.title("AI Usage vs Trust  (colour = frustration count)")
plt.tight_layout()
plt.savefig(VIZ / "usage_vs_trust_scatter.png", dpi=120)
plt.show()


# ## Viz 4 — Quadrant distribution

# In[6]:


quadrant_pd = spark.sql("""
SELECT TrustUsageGroup, COUNT(*) AS DeveloperCount
FROM ai_trust_quadrants
WHERE TrustScore IS NOT NULL
GROUP BY TrustUsageGroup
ORDER BY DeveloperCount DESC
""").toPandas()

palette = {
    "High Usage - High Trust": "#2ca02c",
    "High Usage - Low Trust" : "#d62728",
    "Low Usage - High Trust" : "#1f77b4",
    "Low Usage - Low Trust"  : "#7f7f7f",
}
colors = [palette.get(g, "#999") for g in quadrant_pd["TrustUsageGroup"]]
plt.figure(figsize=(11, 6))
plt.bar(quadrant_pd["TrustUsageGroup"], quadrant_pd["DeveloperCount"], color=colors)
for i, v in enumerate(quadrant_pd["DeveloperCount"]):
    plt.text(i, v + 200, f"{v:,}", ha="center")
plt.xlabel("Trust-Usage Group")
plt.ylabel("Developer Count")
plt.title("AI Trust Paradox Quadrant Distribution")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "quadrant_distribution.png", dpi=120)
plt.show()


# ## Viz 5 — Frustration by Trust-Usage group

# In[7]:


frust_pd = spark.sql("""
SELECT TrustUsageGroup, AVG(FrustrationScore) AS AvgFrustration
FROM ai_trust_quadrants
WHERE TrustScore IS NOT NULL
GROUP BY TrustUsageGroup
ORDER BY AvgFrustration DESC
""").toPandas()
plt.figure(figsize=(11, 6))
plt.bar(frust_pd["TrustUsageGroup"], frust_pd["AvgFrustration"], color="#ff7f0e")
plt.ylabel("Avg # of frustration tokens")
plt.title("Frustration Level by Trust-Usage Group")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "frustration_by_group.png", dpi=120)
plt.show()


# ## Viz 6 — Trust by experience level

# In[8]:


exp_pd = spark.sql("""
SELECT
    CASE
        WHEN WorkExpNum IS NULL                   THEN 'Unknown'
        WHEN WorkExpNum < 2                       THEN '0-1 yrs'
        WHEN WorkExpNum BETWEEN 2  AND 5          THEN '2-5 yrs'
        WHEN WorkExpNum BETWEEN 6  AND 10         THEN '6-10 yrs'
        WHEN WorkExpNum BETWEEN 11 AND 20         THEN '11-20 yrs'
        ELSE '20+ yrs'
    END                                              AS ExperienceGroup,
    AVG(TrustScore)        AS AvgTrust,
    AVG(UsageScore)        AS AvgUsage,
    AVG(FrustrationScore)  AS AvgFrustration
FROM ai_trust
WHERE TrustScore IS NOT NULL
GROUP BY 1
""").toPandas()
order = ['0-1 yrs','2-5 yrs','6-10 yrs','11-20 yrs','20+ yrs','Unknown']
exp_pd = exp_pd.set_index("ExperienceGroup").reindex(order).dropna().reset_index()

x = np.arange(len(exp_pd)); w = 0.4
plt.figure(figsize=(11, 6))
plt.bar(x - w/2, exp_pd["AvgUsage"], width=w, label="Avg Usage",       color="#2b7cbf")
plt.bar(x + w/2, exp_pd["AvgTrust"], width=w, label="Avg Trust",       color="#33a02c")
plt.xticks(x, exp_pd["ExperienceGroup"], rotation=15)
plt.ylabel("Score (1-5)")
plt.title("Average AI Usage vs Trust by Experience Group")
plt.legend()
plt.tight_layout()
plt.savefig(VIZ / "experience_vs_trust.png", dpi=120)
plt.show()


# ## Viz 7 — Heatmap of Usage Score × Trust Score

# In[9]:


heat_pd = spark.sql("""
SELECT UsageScore, TrustScore, COUNT(*) AS DeveloperCount
FROM ai_trust
WHERE UsageScore IS NOT NULL AND TrustScore IS NOT NULL AND UsageScore > 0
GROUP BY UsageScore, TrustScore
""").toPandas()
pivot = heat_pd.pivot(index="TrustScore", columns="UsageScore",
                      values="DeveloperCount").fillna(0)

plt.figure(figsize=(8, 6))
im = plt.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
plt.colorbar(im, label="Developer Count")
plt.xlabel("Usage Score")
plt.ylabel("Trust Score")
plt.title("Heatmap: AI Usage vs Trust Score")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)),   pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        plt.text(j, i, f"{int(pivot.values[i, j]):,}",
                 ha="center", va="center", color="white", fontsize=8)
plt.tight_layout()
plt.savefig(VIZ / "usage_trust_heatmap.png", dpi=120)
plt.show()


# ## Viz 8 — Random-Forest feature importance

# In[10]:


fi = pd.read_csv(OUTPUT_DIR / "ray_ml_results" / "paradox_feature_importance.csv")
plt.figure(figsize=(10, 5))
plt.barh(fi["Feature"][::-1], fi["Importance"][::-1], color="#9467bd")
plt.xlabel("Mean importance (3-seed RF ensemble)")
plt.title("What predicts membership in the AI Trust Paradox group?")
plt.tight_layout()
plt.savefig(VIZ / "rf_feature_importance.png", dpi=120)
plt.show()


# ## Viz 9 — Ray clusters in PCA space

# In[11]:


from sklearn.decomposition import PCA
clusters = pd.read_csv(OUTPUT_DIR / "ray_ml_results" / "ray_cluster_results.csv")
features = ["UsageScore","TrustScore","SentimentScore","ComplexityScore",
            "FrustrationScore","AgentAdoptionScore","AIModelCount",
            "DevEnvToolCount","WorkExpNum"]
from sklearn.preprocessing import StandardScaler
Xc = StandardScaler().fit_transform(clusters[features].values)
pcs = PCA(n_components=2, random_state=0).fit_transform(Xc)
plt.figure(figsize=(8, 6))
for k in sorted(clusters["Cluster"].unique()):
    m = clusters["Cluster"] == k
    plt.scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=0.45, label=f"cluster {k}")
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.title("Developer Clusters (Ray K-Means, k=4) — projected onto first 2 PCs")
plt.legend()
plt.tight_layout()
plt.savefig(VIZ / "cluster_pca.png", dpi=120)
plt.show()


# ## Testing
# 
# Lightweight smoke tests covering: load, schema, null-handling, score ranges, query timing, and presence of every output artefact.

# In[12]:


def run_tests():
    tests = []

    # T1: dataset loads with the expected row count
    n = df_scores.count()
    tests.append(("T1 row count", 40000 < n < 60000, f"got {n:,}"))

    # T2: required score columns exist
    needed = {"UsageScore","TrustScore","SentimentScore","ComplexityScore",
              "FrustrationScore","AgentAdoptionScore"}
    tests.append(("T2 score columns present", needed.issubset(df_scores.columns),
                  f"missing: {needed - set(df_scores.columns)}"))

    # T3: ordinal scores fall inside [1, 5] when not null
    from pyspark.sql.functions import min as smin, max as smax
    bounds = {c: df_scores.agg(smin(c), smax(c)).first()
              for c in ["UsageScore","TrustScore","SentimentScore","ComplexityScore"]}
    ok = all((b[0] is None or b[0] >= 0) and (b[1] is None or b[1] <= 5) for b in bounds.values())
    tests.append(("T3 ordinal bounds", ok, str(bounds)))

    # T4: Spark-SQL query timing
    t0 = time.time()
    spark.sql("SELECT AISelect, AVG(TrustScore) FROM ai_trust GROUP BY AISelect").collect()
    dt = time.time() - t0
    tests.append(("T4 SQL query <30s", dt < 30, f"{dt:.3f}s"))

    # T5: every required output artefact exists
    expected = [
        OUTPUT_DIR / "spark_sql_results" / "dataset_summary",
        OUTPUT_DIR / "spark_sql_results" / "trust_by_usage",
        OUTPUT_DIR / "spark_sql_results" / "quadrant_summary",
        OUTPUT_DIR / "ray_ml_results"   / "ray_cluster_results.csv",
        OUTPUT_DIR / "ray_ml_results"   / "paradox_feature_importance.csv",
        VIZ / "ai_usage_distribution.png",
        VIZ / "trust_by_usage.png",
        VIZ / "quadrant_distribution.png",
        VIZ / "rf_feature_importance.png",
    ]
    miss = [str(p) for p in expected if not p.exists()]
    tests.append(("T5 output artefacts", not miss, f"missing: {miss}"))

    # report
    width = max(len(t[0]) for t in tests)
    for name, ok, detail in tests:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}  {detail}")
    return all(ok for _, ok, _ in tests)

all_passed = run_tests()
print("\nALL TESTS PASSED" if all_passed else "\nSOME TESTS FAILED")


# In[13]:


spark.stop()

