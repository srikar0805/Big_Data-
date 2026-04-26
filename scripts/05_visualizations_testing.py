"""
Phase 2 — Step 05 — Visualisations + smoke tests.

Generates 9 PNGs into output/visualizations/ and runs a small test harness.
All charts use the multi-column **composite** UsageScore / TrustScore /
FrustrationScore (0..100 scale) defined in script 02.

Reads  : output/cleaned_data/ai_trust_scores
         output/ray_ml_results/{paradox_feature_importance.csv,
                                ray_cluster_results.csv}
Writes : output/visualizations/*.png
"""
import os, time, json
from pathlib import Path

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

PROJECT_ROOT = Path("/users/sk7dn/big_data/AI_Trust_Paradox_Phase2")
OUTPUT_DIR   = PROJECT_ROOT / "output"
VIZ          = OUTPUT_DIR / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

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

analysed = df_scores.filter(col("TrustScore").isNotNull())
median_usage = analysed.approxQuantile("UsageScore", [0.5], 0.001)[0]
median_trust = analysed.approxQuantile("TrustScore", [0.5], 0.001)[0]
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

# -----------------------------------------------------------------------------
# Viz 1 — AI usage frequency distribution (raw category from AISelect)
# -----------------------------------------------------------------------------
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
plt.close()

# -----------------------------------------------------------------------------
# Viz 2 — Average composite TrustScore by AI-usage frequency category
# -----------------------------------------------------------------------------
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
plt.ylabel("Average Composite Trust Score (0-100)")
plt.title("Average AI Trust by Usage Frequency")
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "trust_by_usage.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 3 — Composite UsageScore vs TrustScore (jittered scatter, coloured by frustration)
# -----------------------------------------------------------------------------
scatter_pd = (
    df_scores.select("UsageScore", "TrustScore", "FrustrationScore")
             .filter("TrustScore IS NOT NULL")
             .toPandas()
)
rng = np.random.default_rng(0)
sx = scatter_pd["UsageScore"] + rng.uniform(-0.6, 0.6, len(scatter_pd))
sy = scatter_pd["TrustScore"] + rng.uniform(-0.6, 0.6, len(scatter_pd))
plt.figure(figsize=(9, 7))
sc = plt.scatter(sx, sy, c=scatter_pd["FrustrationScore"],
                 alpha=0.30, s=10, cmap="plasma")
plt.colorbar(sc, label="Frustration Score (0-100)")
plt.axvline(median_usage, ls="--", color="black", lw=1, alpha=0.5,
            label=f"median Usage = {median_usage:.1f}")
plt.axhline(median_trust, ls="--", color="black", lw=1, alpha=0.5,
            label=f"median Trust = {median_trust:.1f}")
plt.xlabel("Composite Usage Score (0-100)")
plt.ylabel("Composite Trust Score (0-100)")
plt.title("AI Usage vs Trust  (colour = frustration level)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(VIZ / "usage_vs_trust_scatter.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 4 — Quadrant distribution
# -----------------------------------------------------------------------------
quadrant_pd = spark.sql("""
SELECT TrustUsageGroup, COUNT(*) AS DeveloperCount
FROM ai_trust_quadrants
GROUP BY TrustUsageGroup
ORDER BY DeveloperCount DESC
""").toPandas()

palette = {
    "High Usage - High Trust": "#2ca02c",
    "High Usage - Low Trust" : "#d62728",  # paradox group highlighted in red
    "Low Usage - High Trust" : "#1f77b4",
    "Low Usage - Low Trust"  : "#7f7f7f",
}
colors = [palette.get(g, "#999") for g in quadrant_pd["TrustUsageGroup"]]
plt.figure(figsize=(11, 6))
plt.bar(quadrant_pd["TrustUsageGroup"], quadrant_pd["DeveloperCount"], color=colors)
for i, v in enumerate(quadrant_pd["DeveloperCount"]):
    plt.text(i, v + 100, f"{v:,}", ha="center")
plt.xlabel("Trust-Usage Group  (split on median composite scores)")
plt.ylabel("Developer Count")
plt.title("AI Trust Paradox Quadrant Distribution")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "quadrant_distribution.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 5 — Frustration by trust-usage group
# -----------------------------------------------------------------------------
frust_pd = spark.sql("""
SELECT TrustUsageGroup, AVG(FrustrationScore) AS AvgFrustration
FROM ai_trust_quadrants
GROUP BY TrustUsageGroup
ORDER BY AvgFrustration DESC
""").toPandas()
plt.figure(figsize=(11, 6))
colors = [palette.get(g, "#999") for g in frust_pd["TrustUsageGroup"]]
plt.bar(frust_pd["TrustUsageGroup"], frust_pd["AvgFrustration"], color=colors)
for i, v in enumerate(frust_pd["AvgFrustration"]):
    plt.text(i, v + 0.5, f"{v:.1f}", ha="center")
plt.ylabel("Avg Composite Frustration Score (0-100)")
plt.title("Frustration Level by Trust-Usage Group")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(VIZ / "frustration_by_group.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 6 — Composite Usage vs Trust by experience bracket
# -----------------------------------------------------------------------------
exp_pd = spark.sql("""
SELECT
    CASE
        WHEN WorkExpNum IS NULL              THEN 'Unknown'
        WHEN WorkExpNum < 2                  THEN '0-1 yrs'
        WHEN WorkExpNum BETWEEN 2  AND 5     THEN '2-5 yrs'
        WHEN WorkExpNum BETWEEN 6  AND 10    THEN '6-10 yrs'
        WHEN WorkExpNum BETWEEN 11 AND 20    THEN '11-20 yrs'
        ELSE '20+ yrs'
    END                                       AS ExperienceGroup,
    AVG(UsageScore)        AS AvgUsage,
    AVG(TrustScore)        AS AvgTrust,
    AVG(FrustrationScore)  AS AvgFrustration
FROM ai_trust
WHERE TrustScore IS NOT NULL
GROUP BY 1
""").toPandas()
order = ['0-1 yrs', '2-5 yrs', '6-10 yrs', '11-20 yrs', '20+ yrs', 'Unknown']
exp_pd = exp_pd.set_index("ExperienceGroup").reindex(order).dropna().reset_index()

x = np.arange(len(exp_pd)); w = 0.35
plt.figure(figsize=(11, 6))
plt.bar(x - w/2, exp_pd["AvgUsage"], width=w, label="Avg Usage Score", color="#2b7cbf")
plt.bar(x + w/2, exp_pd["AvgTrust"], width=w, label="Avg Trust Score", color="#33a02c")
plt.xticks(x, exp_pd["ExperienceGroup"], rotation=15)
plt.ylabel("Score (0-100)")
plt.ylim(0, 100)
plt.title("AI Usage vs Trust by Experience Group  (composite scores)")
plt.legend()
plt.tight_layout()
plt.savefig(VIZ / "experience_vs_trust.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 7 — Heatmap of binned UsageScore × TrustScore counts
# -----------------------------------------------------------------------------
df_h = scatter_pd.dropna().copy()
df_h["u_bin"] = (df_h["UsageScore"] // 10).clip(upper=9).astype(int) * 10
df_h["t_bin"] = (df_h["TrustScore"] // 10).clip(upper=9).astype(int) * 10
heat = (df_h.groupby(["t_bin", "u_bin"]).size()
            .unstack(fill_value=0)
            .reindex(index=range(0, 100, 10),
                     columns=range(0, 100, 10), fill_value=0))

plt.figure(figsize=(8, 6.5))
im = plt.imshow(heat.values, aspect="auto", cmap="viridis", origin="lower",
                extent=[0, 100, 0, 100])
plt.colorbar(im, label="Developer Count")
plt.xlabel("Composite Usage Score (binned by 10)")
plt.ylabel("Composite Trust Score (binned by 10)")
plt.title("Heatmap: Composite Usage vs Trust")
plt.axvline(median_usage, ls="--", color="white", lw=1, alpha=0.7)
plt.axhline(median_trust, ls="--", color="white", lw=1, alpha=0.7)
plt.tight_layout()
plt.savefig(VIZ / "usage_trust_heatmap.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 8 — Random-Forest feature importance
# -----------------------------------------------------------------------------
fi = pd.read_csv(OUTPUT_DIR / "ray_ml_results" / "paradox_feature_importance.csv")
plt.figure(figsize=(10, 5))
plt.barh(fi["Feature"][::-1], fi["Importance"][::-1], color="#9467bd")
plt.xlabel("Mean importance (3-seed RF ensemble)")
plt.title("What predicts membership in the AI Trust Paradox group?")
plt.tight_layout()
plt.savefig(VIZ / "rf_feature_importance.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Viz 9 — Ray clusters in PCA space
# -----------------------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
clusters = pd.read_csv(OUTPUT_DIR / "ray_ml_results" / "ray_cluster_results.csv")
cluster_features = [
    "UsageScore", "TrustScore", "FrustrationScore",
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    "ProblemCount", "ThreatLevel",
    "DevEnvToolCount", "WorkExpNum",
]
Xc = StandardScaler().fit_transform(clusters[cluster_features].values)
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
plt.close()

# =============================================================================
# Testing
# =============================================================================
def run_tests():
    tests = []

    n = df_scores.count()
    tests.append(("T1 row count",
                  40000 < n < 60000,
                  f"got {n:,}"))

    needed = {"UsageScore", "TrustScore", "FrustrationScore",
              "UsageFreq", "AgentDepth", "TrustAcc", "TrustComplex",
              "Sentiment", "ProblemCount", "ThreatLevel"}
    tests.append(("T2 score columns present",
                  needed.issubset(df_scores.columns),
                  f"missing: {needed - set(df_scores.columns)}"))

    from pyspark.sql.functions import min as smin, max as smax
    bounds = {c: df_scores.agg(smin(c), smax(c)).first()
              for c in ["UsageScore", "TrustScore", "FrustrationScore"]}
    ok = all((b[0] is None or b[0] >= 0) and (b[1] is None or b[1] <= 100)
             for b in bounds.values())
    tests.append(("T3 composite-score 0-100 bounds", ok, str(bounds)))

    t0 = time.time()
    spark.sql("""
      SELECT TrustUsageGroup, AVG(TrustScore) FROM ai_trust_quadrants
      GROUP BY TrustUsageGroup
    """).collect()
    dt = time.time() - t0
    tests.append(("T4 SQL query <30s", dt < 30, f"{dt:.3f}s"))

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
        VIZ / "usage_vs_trust_scatter.png",
        VIZ / "frustration_by_group.png",
        VIZ / "experience_vs_trust.png",
        VIZ / "usage_trust_heatmap.png",
        VIZ / "cluster_pca.png",
    ]
    miss = [str(p) for p in expected if not p.exists()]
    tests.append(("T5 output artefacts", not miss, f"missing: {miss}"))

    width = max(len(t[0]) for t in tests)
    for name, ok, detail in tests:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}  {detail}")
    return all(ok for _, ok, _ in tests)

print()
all_passed = run_tests()
print("\nALL TESTS PASSED" if all_passed else "\nSOME TESTS FAILED")
spark.stop()
