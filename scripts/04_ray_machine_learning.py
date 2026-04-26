#!/usr/bin/env python
# coding: utf-8

# # Notebook 04 — Ray Machine Learning
# 
# Two ML tasks, both orchestrated through Ray:
# 1. **K-Means clustering** to discover natural usage/trust segments
# 2. **Random-Forest classifier** that predicts whether a developer belongs to the *AI Trust Paradox* group (High Usage + Low Trust)
# 
# Ray is started in single-node `local` mode and runs the sklearn fit calls inside Ray remote tasks, which gives us the parallel-execution story required by the Phase-2 spec while keeping the code drop-in.

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
    .appName("AI Trust Paradox - 04 RayML")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df_scores = spark.read.parquet(str(OUTPUT_DIR / "cleaned_data" / "ai_trust_scores"))
print("rows:", df_scores.count())


# ## 4.1  Add the Paradox label
# 
# Same definition as in notebook 03: paradox = UsageScore ≥ mean **and** TrustScore < mean.

# In[3]:


avg_usage = df_scores.agg(avg("UsageScore")).collect()[0][0]
avg_trust = df_scores.agg(avg("TrustScore")).collect()[0][0]

df_scores = df_scores.withColumn(
    "ParadoxLabel",
    when((col("UsageScore") >= avg_usage) & (col("TrustScore") < avg_trust), 1)
    .otherwise(0)
)
df_scores.groupBy("ParadoxLabel").count().show()


# ## 4.2  Pull the ML matrix into Pandas
# 
# We drop rows missing any of the trust-related ordinal scores; `UsageScore` / `AgentAdoptionScore` / `*Count` / `WorkExpNum` are kept even when 0 because 0 has a real meaning (no usage, no models).

# In[4]:


ml_df = df_scores.select(
    "UsageScore", "TrustScore", "SentimentScore", "ComplexityScore",
    "FrustrationScore", "AgentAdoptionScore", "AIModelCount",
    "DevEnvToolCount", "WorkExpNum", "ParadoxLabel"
).dropna().toPandas()
print("ML rows:", len(ml_df))
ml_df.head()


# ## 4.3  Start Ray (local single-node)

# In[5]:


import ray
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
print(ray.cluster_resources())


# ## 4.4  ML Task 1 — K-Means clustering (executed via Ray)
# 
# We sweep `k = 2..6` in parallel as Ray remote tasks, score each by silhouette + inertia, pick the winner, and label the dataset.

# In[6]:


import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

features = [
    "UsageScore", "TrustScore", "SentimentScore", "ComplexityScore",
    "FrustrationScore", "AgentAdoptionScore", "AIModelCount",
    "DevEnvToolCount", "WorkExpNum",
]
X = ml_df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@ray.remote
def fit_kmeans(X, k, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=5000, random_state=seed) if k > 1 else float("nan")
    return {"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)}

scan = ray.get([fit_kmeans.remote(X_scaled, k) for k in range(2, 7)])
scan_df = pd.DataFrame(scan).sort_values("k").reset_index(drop=True)
print(scan_df)
scan_df.to_csv(OUTPUT_DIR / "ray_ml_results" / "kmeans_scan.csv", index=False)


# In[7]:


# fit final K-Means with k=4 (matches the 4-quadrant story)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
ml_df["Cluster"] = kmeans.fit_predict(X_scaled)
cluster_summary = ml_df.groupby("Cluster")[features].mean().round(3)
cluster_sizes   = ml_df.groupby("Cluster").size().rename("DeveloperCount")
print(cluster_sizes)
print(cluster_summary)
ml_df.to_csv(OUTPUT_DIR / "ray_ml_results" / "ray_cluster_results.csv", index=False)
cluster_summary.assign(DeveloperCount=cluster_sizes).to_csv(
    OUTPUT_DIR / "ray_ml_results" / "cluster_summary.csv")


# ## 4.5  ML Task 2 — Random-Forest classifier for the Paradox group
# 
# Ray runs three independent forests (different `random_state`s) in parallel; we ensemble-average their predictions. This is a tiny demonstration of Ray's parallel ML primitives — the same pattern scales to a real cluster.

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# NB: TrustScore is excluded from the predictors because the label is *defined*
# from it — keeping it would leak the answer.
predictors = [
    "UsageScore", "SentimentScore", "ComplexityScore", "FrustrationScore",
    "AgentAdoptionScore", "AIModelCount", "DevEnvToolCount", "WorkExpNum",
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
        "preds": m.predict(Xte),
        "proba": m.predict_proba(Xte)[:, 1],
        "importances": m.feature_importances_,
    }

results = ray.get([
    fit_forest.remote(X_train, y_train, X_test, seed)
    for seed in (42, 1337, 2025)
])
proba_avg = np.mean([r["proba"] for r in results], axis=0)
preds_ens = (proba_avg >= 0.5).astype(int)

print("Ensemble accuracy:", accuracy_score(y_test, preds_ens))
print(classification_report(y_test, preds_ens, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, preds_ens))


# In[9]:


imp_avg = np.mean([r["importances"] for r in results], axis=0)
feature_importance = (
    pd.DataFrame({"Feature": predictors, "Importance": imp_avg})
      .sort_values("Importance", ascending=False)
      .reset_index(drop=True)
)
print(feature_importance)
feature_importance.to_csv(
    OUTPUT_DIR / "ray_ml_results" / "paradox_feature_importance.csv",
    index=False
)


# ## 4.6  Persist held-out predictions and metrics

# In[10]:


metrics = {
    "accuracy": float(accuracy_score(y_test, preds_ens)),
    "n_train":  int(len(X_train)),
    "n_test":   int(len(X_test)),
    "positive_rate_train": float(y_train.mean()),
    "positive_rate_test":  float(y_test.mean()),
}
import json
(OUTPUT_DIR / "ray_ml_results" / "rf_metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)

predictions_df = X_test.copy()
predictions_df["y_true"] = y_test.values
predictions_df["y_pred"] = preds_ens
predictions_df["y_proba"] = proba_avg
predictions_df.to_csv(OUTPUT_DIR / "ray_ml_results" / "rf_predictions.csv", index=False)


# In[11]:


ray.shutdown()
spark.stop()
print("done")

