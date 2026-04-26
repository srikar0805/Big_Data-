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
output/cleaned_data/ai_trust_scores      (composite UsageScore / TrustScore /
                                          FrustrationScore on 0–100 scale,
                                          plus the per-component sub-scores)
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

## The composite scores (script 02)

Each headline score is built from **multiple survey columns**, normalised to
[0, 1] per component, averaged, and rescaled to **0 – 100** so the three
scores are directly comparable.

### `UsageScore` — built from 4 columns
*"How deeply integrated is this developer with AI tools?"*

| Component | Survey column | Normalisation |
|---|---|---|
| Frequency | `AISelect` | daily=5 / weekly=4 / monthly=3 / plan=2 / no-plan=1 / skip=0  → ÷5 |
| Agent depth | `AIAgents` | daily=5 / weekly=4 / monthly=3 / copilot-only=2 / plan=1 / no=0  → ÷5 |
| Model breadth | `AIModelsHaveWorkedWith` | count of models worked with, capped at 10 → ÷10 |
| Workflow integration | `AIToolCurrently mostly AI` + `AIToolCurrently partially AI` | `mostly + 0.5 × partially`, capped at 10 → ÷10 |

`UsageScore = mean(4 components) × 100`

### `TrustScore` — built from 3 columns
*"How much does this developer trust AI?"*

| Component | Survey column | Normalisation |
|---|---|---|
| Accuracy trust | `AIAcc` | Highly trust=5 … Highly distrust=1 → (x-1)/4 |
| Complex-task trust | `AIComplex` | Very well=5 … Very poor=1 → (x-1)/4 |
| Sentiment | `AISent` | Very favorable=5 … Very unfavorable=1 → (x-1)/4 |

`TrustScore = mean(3 components) × 100` &nbsp; (NULL if any component skipped — 27,258 / 49,191 respondents have a TrustScore)

### `FrustrationScore` — built from 2 columns
*"How frustrated is this developer with AI?"*

| Component | Survey column | Normalisation |
|---|---|---|
| Problem count | `AIFrustration` | # of problem tokens (excluding "haven't encountered" and "don't use regularly"), capped at 5 → ÷5 |
| Threat perception | `AIThreat` | Yes=1.0 / Unsure=0.5 / No=0 |

`FrustrationScore = mean(2 components) × 100`

The per-component scores (`UsageFreq`, `TrustAcc`, `Sentiment`, …) are kept
as separate columns in the parquet output so the report and the ML script
can use them as individual features.

### Quadrant definition
The **AI Trust Paradox group** is `UsageScore ≥ median(UsageScore)` **AND**
`TrustScore < median(TrustScore)`, computed over the 27,258 respondents who
answered all three trust questions. Median is used (not mean) for a robust
50/50 split independent of the long left tail of non-responders.

---

## Headline findings (from this run)

| Usage band | Devs | Avg Trust | Avg Frustration |
|---|---:|---:|---:|
| Low (0–25)  | 25,532 | 30.1 | 9.9 |
| Med-Low (25–50) | 13,855 | 52.6 | 28.4 |
| Med-High (50–75) | 8,103 | 63.7 | 32.1 |
| High (75–100) | 1,701 | 74.4 | 34.9 |

→ Trust **and** frustration both rise with usage. So heavy use generally
*does* build trust, but ~16 % of high-usage developers stay high-usage
*despite* low trust — the paradox group.

| Quadrant (median split) | Devs | Avg Frustration |
|---|---:|---:|
| High Usage – High Trust | 9,480 | 31.3 |
| Low Usage – Low Trust | 8,671 | 26.6 |
| Low Usage – High Trust | 4,795 | 25.7 |
| **High Usage – Low Trust (paradox)** | **4,312** | **33.1** ← highest |

| Predictor of paradox membership | Mean RF importance |
|---|---:|
| `WorkExpNum` | 22.0 % |
| `WorkflowIntegration` | 19.9 % |
| `AIModelCount` | 15.1 % |
| `DevEnvToolCount` | 13.5 % |
| `AgentDepth` | 13.0 % |

Random-forest ensemble accuracy on held-out test set: **83.6 %**.

---

## Testing

`scripts/05_visualizations_testing.py` ships a small smoke-test harness that
checks: (T1) row count after cleaning, (T2) presence of required score
columns, (T3) composite-score 0–100 bounds, (T4) Spark-SQL query latency,
and (T5) existence of every expected output artefact. It prints a
PASS/FAIL line per test plus a final aggregate.
