# Dashboard

Interactive Panel + Plotly dashboard for the AI Trust Paradox analysis.

## What's in it

**Sidebar filters** (apply across every chart):
- Developer role (top-15 multi-select)
- Country (top-15 multi-select)
- Years of work experience (range slider)
- "Show only the paradox group" toggle

**KPI strip**, 6 live cards: total devs analysed, paradox count, paradox %,
mean Usage, mean Trust, mean Frustration.

**Tab, The Paradox**
- Quadrant bar chart (paradox group highlighted in red)
- Usage × Trust scatter (jittered, coloured by frustration, with median lines)
- Top roles in the paradox group

**Tab, Trust Dynamics**
- Trust + Frustration by Usage band
- Composite scores by experience bracket
- Top 15 roles by average trust
- Heatmap of binned Usage × Trust

**Tab, Machine Learning**
- Random-Forest feature importance (with model accuracy)
- K-Means k-sweep curve (inertia + silhouette)
- Cluster centroids table
- PCA projection of the 4 K-Means clusters
- Plain-English explainer of what each model is doing

## Run locally

From the project root:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64    # only needed if Spark scripts haven't been run
panel serve dashboard/app.py --show --autoreload --port 5006
```

`--show` opens your browser at `http://localhost:5006/app`.
`--autoreload` re-loads the dashboard when you edit `app.py`.

## Run inside Microsoft Fabric

Two options.

**Option 1, drop the script into a Fabric notebook**

```python
%pip install panel plotly
```

Then in a fresh cell:

```python
import sys
sys.path.insert(0, "/lakehouse/default/Files/AI_Trust_Paradox_Phase2")
from dashboard.app import dashboard
dashboard.servable()
```

The dashboard renders inline in the notebook output cell.

**Option 2, run as a standalone Bokeh server from a Fabric notebook**

```python
!panel serve /lakehouse/default/Files/AI_Trust_Paradox_Phase2/dashboard/app.py \
    --address 0.0.0.0 --port 5006 --allow-websocket-origin='*' &
```

Then expose port 5006 via Fabric's tunneling / VS Code remote.

## Data dependencies

The dashboard reads from `output/`, make sure scripts 01-04 have been
run at least once before launching:

```
output/cleaned_data/ai_trust_scores/      ← from script 02
output/cleaned_data/ai_trust_quadrants/   ← from script 03
output/ray_ml_results/cluster_summary.csv         ← from script 04
output/ray_ml_results/ray_cluster_results.csv     ← from script 04
output/ray_ml_results/paradox_feature_importance.csv  ← from script 04
output/ray_ml_results/kmeans_scan.csv             ← from script 04
output/ray_ml_results/rf_metrics.json             ← from script 04
```

## Tech stack

- **Panel 1.8**, declarative dashboarding
- **Plotly**, interactive charts (zoom / pan / hover)
- **Tabulator** (panel built-in), interactive cluster centroid table
- **scikit-learn**, `StandardScaler` + `PCA` for the cluster scatter
- **pandas / pyarrow**, read parquet outputs

No callbacks to write, every chart is a `pn.bind(fn, *widgets)` call,
which is Panel's reactive idiom.
