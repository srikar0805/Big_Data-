# AI Trust Paradox, Phase 2

**An Empirical Big Data Study of Trust in AI Coding Tools Using Apache Spark and Ray**

CMP_SC-8540, Big Data and Model Management, Spring 2026
Team: Preya Patel, Sai Srikar

[![pipeline](https://img.shields.io/badge/pipeline-5%2F5%20scripts-success)]()
[![tests](https://img.shields.io/badge/tests-5%2F5%20PASS-success)]()
[![rf-accuracy](https://img.shields.io/badge/RF%20accuracy-83.6%25-blue)]()

---

## Research question

> *Do developers who use AI tools more frequently trust them more, or does
> heavy usage expose limitations and create skepticism?*

We answer this with the 2025 Stack Overflow Developer Survey (49,191
respondents, 172 columns), built three **multi-column composite scores**
(Usage / Trust / Frustration on a 0–100 scale), ran Spark-SQL analytics
to split the population into four trust × usage quadrants, and trained
Ray-orchestrated clustering and classification models to characterise the
**AI Trust Paradox** group, developers who use AI heavily yet still
distrust it. The whole analysis is then exposed via an interactive Panel
dashboard.

---

## Folder structure

```
AI_Trust_Paradox_Phase2/
├── data/                                 # download instructions + schema only
│   ├── README.md                         # how to fetch the 135 MB CSV
│   └── survey_results_schema.csv         # column → question text (30 KB)
├── scripts/                              # Python pipeline (Fabric-ready)
│   ├── 01_data_ingestion_cleaning.py     # CSV → cleaned Parquet
│   ├── 02_feature_engineering_scores.py  # composite Usage / Trust / Frustration
│   ├── 03_spark_sql_analytics.py         # 7 Spark-SQL queries + quadrants
│   ├── 04_ray_machine_learning.py        # Ray K-Means + Ray RF ensemble
│   └── 05_visualizations_testing.py      # 9 PNGs + 5 smoke tests
├── dashboard/                            # Interactive Panel + Plotly app
│   ├── app.py                            # 3-tab dashboard (sidebar filters)
│   └── README.md                         # local + Microsoft Fabric run notes
├── logs/                                 # stdout/stderr captured per script
│   └── 0{1..5}_*.log
├── output/                               # all derived artefacts
│   ├── cleaned_data/             # parquet (Spark), gitignored, regenerable
│   ├── spark_sql_results/        # CSVs of every SQL query
│   ├── ray_ml_results/           # cluster labels + RF importances + metrics
│   └── visualizations/           # PNGs (9 charts)
├── report/                       # final PDF report (added later)
├── README.md
├── requirements.txt
└── .gitignore
```

The 135 MB raw CSV is **not** committed (above GitHub's 100 MB single-file
limit and ODbL-licensed source remains a public download). See
[data/README.md](data/README.md) for one-liner download instructions.

---

## Tools used

| layer | tool | where |
|---|---|---|
| Storage / retrieval | Apache Spark 3.5 (Parquet) | scripts 01–05 |
| Analytics | Spark SQL | script 03 |
| ML orchestration | Ray 2.40 (single-node `local`) | script 04 |
| ML algorithms | scikit-learn (`KMeans`, `RandomForestClassifier`) | script 04 |
| Static plots | matplotlib | script 05 |
| Interactive dashboard | Panel 1.8 + Plotly | `dashboard/app.py` |
| Tabular display | Tabulator (Panel built-in) | dashboard ML tab |

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

### Launching the dashboard

After the pipeline has been run at least once:

```bash
panel serve dashboard/app.py --show --autoreload --port 5006
```

See [dashboard/README.md](dashboard/README.md) for filter details and
Microsoft Fabric instructions.

---

## Pipeline at a glance

```
data/survey_results_public.csv  (135 MB CSV, 49,191 × 172)
        │
        ▼   script 01 , Spark CSV  →  Parquet
output/cleaned_data/cleaned_survey_data
        │
        ▼   script 02 , Spark SQL feature-engineering
output/cleaned_data/ai_trust_scores      (composite UsageScore / TrustScore /
                                          FrustrationScore on 0–100 scale,
                                          plus the per-component sub-scores)
        │
        ├─▶ script 03 , Spark SQL analytics
        │      output/spark_sql_results/{dataset_summary,
        │      ai_usage_distribution, trust_by_usage, trust_by_role,
        │      experience_analysis, quadrant_summary, paradox_profile}
        │
        ├─▶ script 04 , Ray K-Means + Ray RF ensemble
        │      output/ray_ml_results/{ray_cluster_results.csv,
        │      cluster_summary.csv, paradox_feature_importance.csv,
        │      rf_metrics.json, rf_predictions.csv, kmeans_scan.csv}
        │
        ├─▶ script 05 , Visualizations + tests
        │      output/visualizations/*.png
        │
        └─▶ dashboard/app.py , Interactive Panel + Plotly dashboard
               sidebar filters · KPI strip · 3 tabs (Paradox / Trust / ML)
```

---

## The composite scores (script 02)

Each headline score is built from **multiple survey columns**, normalised to
[0, 1] per component, averaged, and rescaled to **0 – 100** so the three
scores are directly comparable.

### `UsageScore`, built from 4 columns
*"How deeply integrated is this developer with AI tools?"*

| Component | Survey column | Normalisation |
|---|---|---|
| Frequency | `AISelect` | daily=5 / weekly=4 / monthly=3 / plan=2 / no-plan=1 / skip=0  → ÷5 |
| Agent depth | `AIAgents` | daily=5 / weekly=4 / monthly=3 / copilot-only=2 / plan=1 / no=0  → ÷5 |
| Model breadth | `AIModelsHaveWorkedWith` | count of models worked with, capped at 10 → ÷10 |
| Workflow integration | `AIToolCurrently mostly AI` + `AIToolCurrently partially AI` | `mostly + 0.5 × partially`, capped at 10 → ÷10 |

`UsageScore = mean(4 components) × 100`

### `TrustScore`, built from 3 columns
*"How much does this developer trust AI?"*

| Component | Survey column | Normalisation |
|---|---|---|
| Accuracy trust | `AIAcc` | Highly trust=5 … Highly distrust=1 → (x-1)/4 |
| Complex-task trust | `AIComplex` | Very well=5 … Very poor=1 → (x-1)/4 |
| Sentiment | `AISent` | Very favorable=5 … Very unfavorable=1 → (x-1)/4 |

`TrustScore = mean(3 components) × 100` &nbsp;(NULL if any component skipped, 27,258 / 49,191 respondents have a TrustScore)

### `FrustrationScore`, built from 2 columns
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
*despite* low trust, the paradox group.

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

## Machine learning, what each model predicts

Two ML models are orchestrated in parallel by Ray (script 04):

### K-Means clustering, *"which tribe does this developer belong to?"*

- **Predicts**: a cluster ID (0–3) for each developer, *unsupervised*.
- **Doesn't see** the paradox label at all, it groups people purely on
  similarity in the 11-feature space (composites + components + experience).
- **Why it matters**: one of the 4 natural tribes lines up with the paradox
  quadrant. That's independent confirmation that the paradox is a *real*
  pattern in the data, not just an artefact of how we drew the median lines.
- **Outputs**: `cluster_summary.csv` (centroids), `ray_cluster_results.csv`
  (per-developer cluster IDs), `kmeans_scan.csv` (k-sweep diagnostics).

### Random Forest classifier, *"will this developer be in the paradox group?"*

- **Predicts**: 1 (paradox) or 0 (not paradox) for each developer, *supervised*.
- **Crucial trick**: trust columns (`AIAcc`, `AIComplex`, `AISent`, and the
  composite `TrustScore`) are **excluded** from the predictors, they define
  the label, so keeping them would leak. The model has to predict paradox
  membership using only **how** developers use AI + how frustrated they are
  + how experienced they are.
- **Why it matters**: 83.6 % accuracy without seeing trust scores means
  usage patterns + experience alone are enough to spot a paradox developer.
  The feature importances answer *who* the paradox developer is.
- **Outputs**: `rf_metrics.json`, `rf_predictions.csv`, `paradox_feature_importance.csv`.

---

## Testing

`scripts/05_visualizations_testing.py` ships a small smoke-test harness that
checks:

| # | Test | Latest result |
|---|---|---|
| T1 | Row count after cleaning | PASS · 49,191 |
| T2 | All required score columns present | PASS · 0 missing |
| T3 | Composite scores stay inside [0, 100] | PASS |
| T4 | Spark-SQL aggregate latency < 30 s | PASS · 0.36 s |
| T5 | Every expected output artefact exists | PASS · 0 missing |

It prints a PASS/FAIL line per test plus a final aggregate. See
[logs/05_visualizations_testing.log](logs/05_visualizations_testing.log)
for the latest run.

---

## Dashboard preview

After running the pipeline, launch:

```bash
panel serve dashboard/app.py --show --autoreload --port 5006
```

The dashboard exposes:

| | |
|---|---|
| **Sidebar filters** | DevType (top-15 multi-select) · Country (top-15) · WorkExp range slider · "show only paradox group" toggle |
| **KPI strip** | live counts and mean composite scores (re-render under filters) |
| **Tab 1 · The Paradox** | quadrant bar chart · jittered Usage × Trust scatter with median lines · top paradox-group roles |
| **Tab 2 · Trust Dynamics** | Trust+Frustration by usage band · scores by experience · top roles by trust · binned heatmap |
| **Tab 3 · Machine Learning** | RF feature importance · K-Means k-sweep curve · cluster centroids table · clusters in PCA space · plain-English explainer |

For Microsoft Fabric instructions, see
[dashboard/README.md](dashboard/README.md).

---

## Team & responsibilities

| Member | Owned |
|---|---|
| **Preya Patel** | Data preprocessing & cleaning · feature-engineering composites · matplotlib visualisations |
| **Sai Srikar** | Spark-SQL quadrant + correlation analysis · Ray ML pipeline · Panel dashboard |

---

## License

Code: MIT. Dataset: ODbL (Open Database License), see
<https://survey.stackoverflow.co/>.
