# AI Trust Paradox, Phase 2

**An Empirical Big Data Study of Trust in AI Coding Tools Using Apache Spark and Ray**

CMP_SC-8540, Big Data and Model Management, Spring 2026
Team: Preya Patel, Sai Srikar

[![pipeline](https://img.shields.io/badge/pipeline-5%2F5%20scripts-success)]()
[![tests](https://img.shields.io/badge/tests-6%2F6%20PASS-success)]()
[![rf-roc-auc](https://img.shields.io/badge/RF%20ROC--AUC-0.837-blue)]()

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
distrust it. The whole analysis is then exposed via a Panel dashboard.

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
│   ├── 05_visualizations_testing.py      # 11 PNGs + 6 smoke tests
│   ├── 00_verify_distributed_runtime.py  # Spark + Ray local runtime check
│   └── runtime_env.py                    # Java/Spark localhost setup helper
├── dashboard/                            # Panel dashboard
│   ├── app.py                            # 3-tab data-backed dashboard
│   └── README.md                         # local + Microsoft Fabric run notes
├── logs/                                 # stdout/stderr captured per script
│   └── 0{1..5}_*.log
├── output/                               # all derived artefacts
│   ├── cleaned_data/             # parquet (Spark), gitignored, regenerable
│   ├── spark_sql_results/        # CSVs of every SQL query
│   ├── ray_ml_results/           # cluster labels + RF importances + metrics
│   └── visualizations/           # PNGs (11 charts)
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
| ML orchestration | Ray 2.x (single-node `local`; Python 3.8 uses Ray 2.10) | script 04 |
| ML algorithms | scikit-learn (`KMeans`, `RandomForestClassifier`) | script 04 |
| Static plots | matplotlib | script 05 |
| Dashboard | Panel 1.8 + Bokeh | `dashboard/app.py` |
| Tabular display | Tabulator (Panel built-in) | dashboard ML tab |

---

## How to run

```bash
# one-time setup
sudo apt-get install -y openjdk-17-jdk-headless python3-venv
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Spark needs Java at JVM start time. Scripts auto-detect either
# JAVA_HOME, ./.jdk, ~/.local/jdk-17, or /usr/lib/jvm/java-17-openjdk-amd64.
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# optional quick proof that Spark and Ray can execute local distributed jobs
python scripts/00_verify_distributed_runtime.py

# run the pipeline (must be in this order)
mkdir -p logs
for s in scripts/0*.py; do
    name=$(basename "$s" .py)
    python "$s" > "logs/${name}.log" 2>&1
done
```

Each script is self-contained (configures Java/Spark localhost defaults,
opens its own Spark session, and resolves paths from the checked-out project
root) so they also run straight inside Microsoft Fabric, Databricks, or any
local Python env.

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
        │      output/visualizations/*.png, generated from Spark/Ray outputs
        │
        └─▶ dashboard/app.py , Panel dashboard
               data-backed KPI strip · Overview · Trust Dynamics · Ray ML Segments
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

### Role, experience, and country profile

Role explains trust differences much more clearly than years of experience.
Among roles with at least 100 developers, **AI app developers** have the
highest average trust (**66.2**), followed by **AI/ML engineers** (**63.4**).
The lowest-trust roles are **game/graphics developers** (**40.6**) and
**embedded developers** (**43.8**).

Experience is almost flat: known-experience groups range only from **52.7**
to **53.4** average trust, a **0.7-point spread**. Including respondents with
unknown experience lowers the minimum to **51.3**, but the overall movement is
still small compared with the role-level differences.

The paradox profile output also supports a country/work-mode view. In the
profile slices with at least 20 developers, the largest segment is
**full-stack developers in the United States working remotely** (131),
followed by **back-end developers in the United States working remotely** (64)
and **full-stack developers in the United States in hybrid work** (50). By
country across those profile slices, the United States leads, followed by
Germany.

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

| Cluster | Segment name | What it means | Developers |
|---:|---|---|---:|
| 0 | Power users | High usage (**67.8**) and high trust (**67.4**), broad model/tool use, and relatively low frustration (**24.8**). | 5,287 |
| 1 | Frustrated adopters | Active AI users with moderate trust (**57.8**) but the highest frustration (**60.4**) and strongest threat concern. | 5,270 |
| 2 | Regular low-friction users | Largest segment: regular AI use, moderate trust (**57.2**), low frustration (**17.1**), and lighter workflow integration. | 9,571 |
| 3 | Low-adoption skeptics | Low usage (**16.9**) and lowest trust (**26.1**); AI is not deeply integrated into their work. | 5,186 |

The PCA cluster chart is a **visual projection**, not the model itself. Ray
K-Means clusters developers in the full 11-feature space; PCA compresses those
features into two axes so we can inspect whether the learned segments separate
visibly. Nearby dots are developers with similar standardized usage/trust/tool
profiles; the PC1/PC2 axes are mixtures of features, not standalone survey
questions.

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

| # | Test | Expected condition |
|---|---|---|
| T1 | Row count after cleaning | 49,191 rows, within the expected survey range |
| T2 | Expected Spark/Ray result columns present | 0 missing required columns |
| T3 | Composite and aggregate scores stay inside [0, 100] | all score fields bounded |
| T4 | Role trust ordering matches the data | AI app developers highest; game/graphics lowest |
| T5 | Known-experience trust range stays under 1 point | confirms experience is nearly flat |
| T6 | Every expected output artefact exists | all 11 visualisation PNGs present |

It prints a PASS/FAIL line per test plus a final aggregate when run in an
environment with the Python plotting dependencies installed.

---

## Dashboard preview

After running the pipeline, launch:

```bash
panel serve dashboard/app.py --address 0.0.0.0 --port 5006 --allow-websocket-origin='*'
```

In the Fabric Test Bed, the dashboard can also render directly inside a
Fabric notebook with `dashboard.servable()`. See [dashboard/README.md](dashboard/README.md)
for both Fabric notebook and standalone Panel/Bokeh options.

The dashboard exposes only charts supported by the available output files:

| | |
|---|---|
| **KPI strip** | survey responses · analysed developers · high-usage/low-trust count/share · mean composite scores · country/role counts |
| **Overview** | quadrant sizes · trust/frustration by usage band |
| **Trust Dynamics** | zoomed trust-by-experience trend · AI usage distribution · highest/lowest role trust |
| **Ray ML Segments** | cluster explanation cards · PCA cluster projection · baseline-aware RF metrics · RF feature importance · K-Means k-sweep · cluster centroids table · model summary |
| **Filtered Profile** | optional row-level role/country filters and charts, shown only when regenerated Parquet exists |

For Microsoft Fabric instructions, see
[dashboard/README.md](dashboard/README.md).

---

## Team & responsibilities

| Member | Owned |
|---|---|
| **Preya Patel** | Data preprocessing & feature-engineering, Visualisations |
| **Sai Srikar** | Correlation analysis, Ray ML pipeline, Panel dashboard |

---

## License

Code: MIT. Dataset: ODbL (Open Database License), see
<https://survey.stackoverflow.co/>.
