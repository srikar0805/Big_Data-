# AI Trust Paradox — Phase 2

**An Empirical Big Data Study of Trust in AI Coding Tools Using Apache Spark and Ray**

CMP_SC-8540 — Big Data and Model Management — Spring 2026
Team: Preya Patel, Sai Srikar

---

## Research question

> Do developers who use AI tools more frequently trust them more, or does
> heavy usage expose limitations and create skepticism?

We answer this with the 2025 Stack Overflow Developer Survey (49,191
respondents, 172 columns), built three composite scores (Usage / Trust /
Frustration), ran Spark-SQL analytics, identified the **AI Trust Paradox**
group (high-usage + low-trust developers), and trained Ray-orchestrated
clustering and classification models to characterise that group.

---

## Folder structure

```
AI_Trust_Paradox_Phase2/
├── data/                         # symlinks into ../data
│   ├── survey_results_public.csv         (135 MB, 49,191 × 172)
│   ├── survey_results_schema.csv         (column → question text)
│   └── 2025_Developer_Survey_Tool.pdf    (the questionnaire)
├── scripts/                              # Python scripts (Fabric-ready)
│   ├── 01_data_ingestion_cleaning.py
│   ├── 02_feature_engineering_scores.py
│   ├── 03_spark_sql_analytics.py
│   ├── 04_ray_machine_learning.py
│   └── 05_visualizations_testing.py
├── logs/                                 # stdout/stderr from each run
│   └── 0{1..5}_*.log
├── output/
│   ├── cleaned_data/             # parquet (Spark) — gitignored, regenerable
│   ├── spark_sql_results/        # CSVs of every SQL query
│   ├── ray_ml_results/           # cluster labels + RF importances + metrics
│   └── visualizations/           # PNGs (9 charts)
├── report/
│   └── Phase2_Report.pdf
├── README.md
└── requirements.txt
```

If you receive the project without `data/survey_results_public.csv`, download
the 2025 dataset from <https://survey.stackoverflow.co/> and place the CSV at
`data/survey_results_public.csv`.

---

## Tools used

| layer | tool |
|---|---|
| Storage / retrieval | Apache Spark 3.5 (Parquet) |
| Analytics | Spark SQL |
| ML orchestration | Ray 2.40 (single-node `local`) |
| ML algorithms | scikit-learn (KMeans, RandomForest) |
| Plotting | matplotlib |

---

## How to run

```bash
# one-time setup
sudo apt-get install -y openjdk-17-jdk-headless python3-venv
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Spark needs JAVA_HOME at JVM start time
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# run the pipeline (must be in this order)
mkdir -p logs
for s in scripts/0*.py; do
    name=$(basename "$s" .py)
    python "$s" > "logs/${name}.log" 2>&1
done
```

Each script is self-contained (sets `JAVA_HOME`, opens its own Spark
session, reads/writes via absolute project-root paths) so they also run
straight inside Microsoft Fabric, Databricks, or any local Python env.

---

## Pipeline at a glance

```
data/survey_results_public.csv  (135 MB CSV, 49,191 × 172)
        │
        ▼   script 01  — Spark CSV  →  Parquet
output/cleaned_data/cleaned_survey_data
        │
        ▼   script 02  — Spark SQL feature-engineering
output/cleaned_data/ai_trust_scores      (+ 10 numeric scores)
        │
        ├─▶ script 03  — Spark SQL analytics
        │      output/spark_sql_results/{dataset_summary,
        │      ai_usage_distribution, trust_by_usage, trust_by_role,
        │      experience_analysis, quadrant_summary, paradox_profile}
        │
        ├─▶ script 04  — Ray K-Means + Ray RF ensemble
        │      output/ray_ml_results/{ray_cluster_results.csv,
        │      cluster_summary.csv, paradox_feature_importance.csv,
        │      rf_metrics.json, rf_predictions.csv, kmeans_scan.csv}
        │
        └─▶ script 05  — Visualizations + tests
               output/visualizations/*.png
```

---

## The composite scores (notebook 02)

| Score | Source column(s) | Range | Meaning |
|---|---|---|---|
| `UsageScore` | `AISelect` | 0–5 | how frequently developer uses AI |
| `TrustScore` | `AIAcc` | 1–5 | trust in AI output accuracy |
| `SentimentScore` | `AISent` | 1–5 | favorability toward AI tools |
| `ComplexityScore` | `AIComplex` | 1–5 | AI quality on complex tasks |
| `FrustrationScore` | `AIFrustration` | int ≥ 0 | # problem tokens picked |
| `AgentAdoptionScore` | `AIAgents` | 0–5 | depth of AI agent adoption |
| `AIModelCount` | `AIModelsHaveWorkedWith` | int ≥ 0 | # AI models worked with |
| `DevEnvToolCount` | `DevEnvsHaveWorkedWith` | int ≥ 0 | # IDEs / dev environments |
| `OverallTrustScore` | composite | 1–5 | mean of trust + sentiment + complexity |
| `OverallUsageScore` | composite | int | usage + agent + model count |

The **AI Trust Paradox group** is defined as
`UsageScore ≥ mean(UsageScore)` AND `TrustScore < mean(TrustScore)`.

---

## Testing

`scripts/05_visualizations_testing.py` ships a small smoke-test harness that
checks: (T1) row count after cleaning, (T2) presence of required score
columns, (T3) ordinal score bounds, (T4) Spark-SQL query latency, and (T5)
existence of every expected output artefact. The notebook prints a
PASS/FAIL line per test plus a final aggregate.
