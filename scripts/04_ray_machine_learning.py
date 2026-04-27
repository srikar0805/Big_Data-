"""
Phase 2 — Step 04 — Ray-orchestrated machine learning.

Two ML tasks run on the multi-column composite scores:

  1.  K-Means clustering — Ray sweeps k=2..6 in parallel remote tasks,
      we score each k by silhouette + inertia and fit a final k=4 model
      to label every developer with a usage/trust segment.

  2.  Random-Forest classifier for the AI Trust Paradox group — three
      forests with different seeds run as Ray remote tasks, we average
      their probabilities for an ensemble prediction. Critically the
      trust components (TrustAcc / TrustComplex / Sentiment / TrustScore)
      are excluded from predictors because the label is *defined* from
      them — keeping them would leak the answer.

Predictors used:
  - UsageFreq, AgentDepth, AIModelCount, WorkflowIntegration  (usage signals)
  - ProblemCount, ThreatLevel                                 (frustration signals)
  - DevEnvToolCount, WorkExpNum                               (developer profile)

Reads  : output/cleaned_data/ai_trust_scores
Writes : output/ray_ml_results/{kmeans_scan, ray_cluster_results,
                                cluster_summary, paradox_feature_importance,
                                rf_metrics, rf_predictions}
"""
from runtime_env import configure_spark_runtime, project_root

PROJECT_ROOT = project_root(__file__)
configure_spark_runtime(PROJECT_ROOT)
OUTPUT_DIR   = PROJECT_ROOT / "output"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 04 RayML")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df_scores = spark.read.parquet(str(OUTPUT_DIR / "cleaned_data" / "ai_trust_scores"))
print("rows:", df_scores.count())

# =============================================================================
# 4.1  Add the Paradox label using the same (median) split as script 03
# =============================================================================
analysed = df_scores.filter(col("TrustScore").isNotNull())
median_usage = analysed.approxQuantile("UsageScore", [0.5], 0.001)[0]
median_trust = analysed.approxQuantile("TrustScore", [0.5], 0.001)[0]
print(f"median UsageScore: {median_usage:.2f}   median TrustScore: {median_trust:.2f}")

analysed = analysed.withColumn(
    "ParadoxLabel",
    when((col("UsageScore") >= median_usage) & (col("TrustScore") < median_trust), 1)
    .otherwise(0)
)
print("\nParadox label distribution:")
analysed.groupBy("ParadoxLabel").count().show()

# =============================================================================
# 4.2  Pull the ML matrix into Pandas
# =============================================================================
ml_df = analysed.select(
    # composite (used for clustering, NOT prediction)
    "UsageScore", "TrustScore", "FrustrationScore",
    # individual usage signals (predictors + cluster features)
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    # individual frustration signals
    "ProblemCount", "ThreatLevel",
    # profile
    "DevEnvToolCount", "WorkExpNum",
    # label
    "ParadoxLabel",
).dropna().toPandas()
print("ML rows:", len(ml_df))
print(ml_df.head())

# =============================================================================
# 4.3  Start Ray (single-node, parallel CPU tasks)
# =============================================================================
import ray
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
print("\nray cluster:", ray.cluster_resources())

# =============================================================================
# 4.4  K-Means clustering (sweep k in parallel via Ray)
# =============================================================================
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cluster_features = [
    "UsageScore", "TrustScore", "FrustrationScore",
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    "ProblemCount", "ThreatLevel",
    "DevEnvToolCount", "WorkExpNum",
]
X = ml_df[cluster_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@ray.remote
def fit_kmeans(X, k, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    sil = (silhouette_score(X, labels, sample_size=5000, random_state=seed)
           if k > 1 else float("nan"))
    return {"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)}

scan = ray.get([fit_kmeans.remote(X_scaled, k) for k in range(2, 7)])
scan_df = pd.DataFrame(scan).sort_values("k").reset_index(drop=True)
print("\nK-Means scan:\n", scan_df)
scan_df.to_csv(OUTPUT_DIR / "ray_ml_results" / "kmeans_scan.csv", index=False)

# Final fit at k=4 (matches the 4-quadrant story)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
ml_df["Cluster"] = kmeans.fit_predict(X_scaled)
cluster_summary = ml_df.groupby("Cluster")[cluster_features].mean().round(3)
cluster_sizes   = ml_df.groupby("Cluster").size().rename("DeveloperCount")
print("\nCluster sizes:\n", cluster_sizes)
print("\nCluster centroids:\n", cluster_summary)
ml_df.to_csv(OUTPUT_DIR / "ray_ml_results" / "ray_cluster_results.csv", index=False)
cluster_summary.assign(DeveloperCount=cluster_sizes).to_csv(
    OUTPUT_DIR / "ray_ml_results" / "cluster_summary.csv")

# =============================================================================
# 4.5  Random-Forest ensemble (parallel forests via Ray)
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Trust columns are EXCLUDED — they define the label and would leak.
predictors = [
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    "ProblemCount", "ThreatLevel",
    "DevEnvToolCount", "WorkExpNum",
]
X = ml_df[predictors]
y = ml_df["ParadoxLabel"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

@ray.remote
def fit_forest(Xtr, ytr, Xte, seed):
    m = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    m.fit(Xtr, ytr)
    return {
        "preds":       m.predict(Xte),
        "proba":       m.predict_proba(Xte)[:, 1],
        "importances": m.feature_importances_,
    }

results = ray.get([fit_forest.remote(X_train, y_train, X_test, s)
                   for s in (42, 1337, 2025)])
proba_avg = np.mean([r["proba"] for r in results], axis=0)
preds_ens = (proba_avg >= 0.5).astype(int)

print("\nEnsemble accuracy:", accuracy_score(y_test, preds_ens))
print(classification_report(y_test, preds_ens, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, preds_ens))

imp_avg = np.mean([r["importances"] for r in results], axis=0)
feature_importance = (
    pd.DataFrame({"Feature": predictors, "Importance": imp_avg})
      .sort_values("Importance", ascending=False)
      .reset_index(drop=True)
)
print("\nFeature importance:\n", feature_importance)
feature_importance.to_csv(
    OUTPUT_DIR / "ray_ml_results" / "paradox_feature_importance.csv", index=False)

# =============================================================================
# 4.6  Persist metrics + held-out predictions
# =============================================================================
import json
metrics = {
    "accuracy"           : float(accuracy_score(y_test, preds_ens)),
    "n_train"            : int(len(X_train)),
    "n_test"             : int(len(X_test)),
    "positive_rate_train": float(y_train.mean()),
    "positive_rate_test" : float(y_test.mean()),
    "median_usage"       : float(median_usage),
    "median_trust"       : float(median_trust),
    "predictors"         : predictors,
}
(OUTPUT_DIR / "ray_ml_results" / "rf_metrics.json").write_text(
    json.dumps(metrics, indent=2))
print("\nmetrics:", metrics)

predictions_df = X_test.copy()
predictions_df["y_true"]  = y_test.values
predictions_df["y_pred"]  = preds_ens
predictions_df["y_proba"] = proba_avg
predictions_df.to_csv(OUTPUT_DIR / "ray_ml_results" / "rf_predictions.csv", index=False)

ray.shutdown()
spark.stop()
print("done")
