# Dashboard

Panel dashboard for the AI Trust Paradox analysis.

## What's in it

The dashboard is intentionally data-backed: it only shows charts supported by
the output files that are present. With the committed Spark-SQL and Ray summary
outputs, it uses native interactive Bokeh charts served by Panel. If the full
pipeline regenerates `output/cleaned_data/` row-level Parquet files, a filtered
profile tab appears with live role, country, and work-experience controls.

**KPI strip**: survey responses, analysed developers, high-usage/low-trust
count and share, mean Usage, mean Trust, mean Frustration, country count, and
developer-role count.

**Tab, Overview**
- Quadrant bar chart (paradox group highlighted in red)
- Trust + Frustration by Usage band

**Tab, Trust Dynamics**
- Focused trust trend by experience bracket
- AI usage frequency distribution
- Roles with the highest and lowest average trust

**Tab, Ray ML Segments**
- Cluster explanation cards with segment names and metrics
- PCA projection of the 4 K-Means clusters
- Baseline-aware classifier metrics: ROC-AUC, balanced accuracy, recall,
  precision, and accuracy vs the always-negative baseline
- Random-Forest feature importance
- K-Means k-sweep curve (inertia + silhouette)
- Cluster profile table with segment names and meanings
- Compact model summary

**Optional tab, Filtered Profile**
- Appears only when row-level Parquet files exist
- Filtered KPI strip and trust-usage quadrants
- Filtered high-usage/low-trust role and country charts

Cluster meanings:

| Cluster | Segment | Short interpretation |
|---:|---|---|
| 0 | Power users | High usage, high trust, broad AI workflow integration. |
| 1 | Frustrated adopters | Active AI use but highest frustration and threat concern. |
| 2 | Regular low-friction users | Largest group; regular AI use with low frustration. |
| 3 | Low-adoption skeptics | Low usage and lowest trust. |

## Run in the Fabric Test Bed

Two supported options.

**Option 1, render inside a Fabric notebook**

```python
%pip install panel bokeh pyarrow scikit-learn
```

Then in a fresh cell:

```python
import sys
sys.path.insert(0, "/lakehouse/default/Files/AI_Trust_Paradox_Phase2")
from dashboard.app import dashboard
dashboard.servable()
```

The dashboard renders inline in the notebook output cell.

**Option 2, run as a standalone Panel/Bokeh app from Fabric**

```python
!panel serve /lakehouse/default/Files/AI_Trust_Paradox_Phase2/dashboard/app.py \
    --address 0.0.0.0 --port 5006 --allow-websocket-origin='*' &
```

Then expose port 5006 through the Fabric Test Bed's supported forwarding or
workspace web endpoint.

## Run locally for development

From the project root on your own machine:

```bash
panel serve dashboard/app.py --show --autoreload --port 5006
```

## Data modes

The dashboard always reads the committed derived outputs:

```
output/spark_sql_results/*                        ← from script 03
output/ray_ml_results/cluster_summary.csv         ← from script 04
output/ray_ml_results/ray_cluster_results.csv     ← from script 04
output/ray_ml_results/paradox_feature_importance.csv  ← from script 04
output/ray_ml_results/kmeans_scan.csv             ← from script 04
output/ray_ml_results/rf_metrics.json             ← from script 04
```

If available, it also loads row-level Parquet and enables role/country/work
experience filters:

```
output/cleaned_data/ai_trust_scores/      ← from script 02
output/cleaned_data/ai_trust_quadrants/   ← from script 03
```

## Tech stack

- **Panel**, declarative dashboarding
- **Bokeh**, native interactive charts with hover, zoom, pan, reset, and save
- **Tabulator** (panel built-in), interactive cluster centroid table
- **scikit-learn**, `StandardScaler` + `PCA` for the cluster scatter
- **pandas / pyarrow**, read CSV and optional Parquet outputs
