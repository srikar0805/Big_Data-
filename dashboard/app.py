"""
AI Trust Paradox, Interactive Dashboard (Panel + Plotly)
=========================================================

Reads the parquet / CSV / JSON artefacts produced by scripts 01-05 and
serves an interactive dashboard with KPIs, sidebar filters, and three
tabs: The Paradox · Trust Dynamics · Machine Learning.

Run locally:
    cd /users/sk7dn/big_data/AI_Trust_Paradox_Phase2
    panel serve dashboard/app.py --show --autoreload --port 5006

Run inside a Microsoft Fabric / Jupyter notebook:
    from dashboard.app import dashboard
    dashboard.servable()                 # Fabric notebook
    # or just:
    dashboard                            # Classic Jupyter
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT       = PROJECT_ROOT / "output"
ML           = OUTPUT / "ray_ml_results"
CLEANED      = OUTPUT / "cleaned_data"

# -----------------------------------------------------------------------------
# Load all the analysis outputs once
# -----------------------------------------------------------------------------
print("loading parquet/CSV outputs ...")
scores    = pd.read_parquet(CLEANED / "ai_trust_scores")
quadrants = pd.read_parquet(CLEANED / "ai_trust_quadrants")

cluster_summary = pd.read_csv(ML / "cluster_summary.csv")
ray_clusters    = pd.read_csv(ML / "ray_cluster_results.csv")
rf_imp          = pd.read_csv(ML / "paradox_feature_importance.csv")
kmeans_scan     = pd.read_csv(ML / "kmeans_scan.csv")
rf_metrics      = json.loads((ML / "rf_metrics.json").read_text())

print(f"  scores:        {len(scores):,} rows")
print(f"  quadrants:     {len(quadrants):,} rows (analysed pop., TrustScore not null)")
print(f"  ray clusters:  {len(ray_clusters):,} rows")

PALETTE = {
    "High Usage - High Trust": "#2ca02c",
    "High Usage - Low Trust" : "#d62728",   # paradox group, red
    "Low Usage - High Trust" : "#1f77b4",
    "Low Usage - Low Trust"  : "#7f7f7f",
}

# Top-N picklists for the sidebar (avoids a 178-country dropdown)
TOP_DEVTYPES  = quadrants["DevType"].value_counts().head(15).index.tolist()
TOP_COUNTRIES = quadrants["Country"].value_counts().head(15).index.tolist()

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
devtype_filter = pn.widgets.MultiChoice(
    name="Developer role",
    options=sorted(TOP_DEVTYPES),
    value=[],
    placeholder="(all roles)",
)
country_filter = pn.widgets.MultiChoice(
    name="Country",
    options=sorted(TOP_COUNTRIES),
    value=[],
    placeholder="(all countries)",
)
exp_filter = pn.widgets.RangeSlider(
    name="Years of work experience",
    start=0, end=50, value=(0, 50), step=1,
)
paradox_only = pn.widgets.Checkbox(
    name="Show only the paradox group", value=False,
)

# -----------------------------------------------------------------------------
# Filter helper
# -----------------------------------------------------------------------------
def filter_df(df, devtypes, countries, exp_range, p_only):
    f = df
    if devtypes:
        f = f[f["DevType"].isin(devtypes)]
    if countries:
        f = f[f["Country"].isin(countries)]
    lo, hi = exp_range
    f = f[(f["WorkExpNum"].fillna(-1) >= lo) & (f["WorkExpNum"].fillna(999) <= hi)]
    if p_only:
        f = f[f["TrustUsageGroup"] == "High Usage - Low Trust"]
    return f

# -----------------------------------------------------------------------------
# KPI cards
# -----------------------------------------------------------------------------
def kpi_card(title, value, fmt="{:,}", color="#1f77b4"):
    return pn.pane.HTML(
        f"""
        <div style="background:#f8f9fa;border-left:4px solid {color};
                    padding:10px 16px;border-radius:6px;height:88px;">
          <div style="color:#666;font-size:11px;text-transform:uppercase;
                      letter-spacing:0.5px;">{title}</div>
          <div style="color:#222;font-size:26px;font-weight:600;
                      margin-top:4px;">{fmt.format(value)}</div>
        </div>
        """,
        margin=(0, 5),
    )

def kpis(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only)
    n = len(f)
    n_paradox = int((f["TrustUsageGroup"] == "High Usage - Low Trust").sum())
    pct_par = (n_paradox / n * 100) if n else 0.0
    return pn.Row(
        kpi_card("Developers analysed", n, "{:,}"),
        kpi_card("Paradox group", n_paradox, "{:,}", color="#d62728"),
        kpi_card("Paradox share", pct_par, "{:.1f}%", color="#d62728"),
        kpi_card("Avg Usage", f["UsageScore"].mean() if n else 0, "{:.1f}"),
        kpi_card("Avg Trust", f["TrustScore"].mean() if n else 0, "{:.1f}", color="#2ca02c"),
        kpi_card("Avg Frustration", f["FrustrationScore"].mean() if n else 0, "{:.1f}", color="#ff7f0e"),
    )

kpi_panel = pn.bind(kpis, devtype_filter, country_filter, exp_filter, paradox_only)

# -----------------------------------------------------------------------------
# Tab 1, The Paradox
# -----------------------------------------------------------------------------
def quadrant_chart(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only)
    counts = (f.groupby("TrustUsageGroup", as_index=False)
                .size().rename(columns={"size": "Developers"}))
    fig = px.bar(
        counts, x="TrustUsageGroup", y="Developers",
        color="TrustUsageGroup", color_discrete_map=PALETTE,
        text="Developers",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        title="Quadrant distribution (filtered)",
        showlegend=False, height=380, margin=dict(t=50, b=50, l=10, r=10),
        xaxis_title=None, yaxis_title="Developer Count",
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

def scatter_chart(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only).copy()
    if len(f) > 6000:
        f = f.sample(6000, random_state=0)
    rng = np.random.default_rng(0)
    f["u_jit"] = f["UsageScore"] + rng.uniform(-0.6, 0.6, len(f))
    f["t_jit"] = f["TrustScore"] + rng.uniform(-0.6, 0.6, len(f))
    fig = px.scatter(
        f, x="u_jit", y="t_jit", color="FrustrationScore",
        color_continuous_scale="plasma",
        opacity=0.55, hover_data={"DevType": True, "WorkExpNum": True,
                                  "u_jit": False, "t_jit": False,
                                  "UsageScore": ":.1f", "TrustScore": ":.1f"},
    )
    median_usage = quadrants["UsageScore"].median()
    median_trust = quadrants["TrustScore"].median()
    fig.add_vline(x=median_usage, line_dash="dash", line_color="black",
                  opacity=0.4, annotation_text=f"median U={median_usage:.1f}")
    fig.add_hline(y=median_trust, line_dash="dash", line_color="black",
                  opacity=0.4, annotation_text=f"median T={median_trust:.1f}")
    fig.update_layout(
        title="Composite Usage vs Trust  (colour = frustration, sample of 6k)",
        xaxis_title="Usage Score", yaxis_title="Trust Score",
        height=480, margin=dict(t=50, b=40, l=10, r=10),
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

def paradox_top_roles(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only).copy()
    f = f[f["TrustUsageGroup"] == "High Usage - Low Trust"]
    s = (f.groupby("DevType")
          .agg(N=("ResponseId", "count"),
               AvgFrustration=("FrustrationScore", "mean"),
               AvgExp=("WorkExpNum", "mean"))
          .reset_index())
    s = s[s["N"] >= 20].sort_values("N", ascending=False).head(12)
    if s.empty:
        return pn.pane.Markdown("*No paradox-group rows under the current filters.*")
    fig = px.bar(
        s.sort_values("N"),
        y="DevType", x="N", orientation="h",
        hover_data={"AvgFrustration": ":.1f", "AvgExp": ":.1f"},
        color="AvgFrustration", color_continuous_scale="plasma",
    )
    fig.update_layout(
        title="Who is in the paradox group?  (top roles by # paradox devs)",
        xaxis_title="Paradox developers", yaxis_title=None,
        height=420, margin=dict(t=50, b=40, l=10, r=10),
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

paradox_tab = pn.Column(
    pn.bind(quadrant_chart,    devtype_filter, country_filter, exp_filter, paradox_only),
    pn.bind(scatter_chart,     devtype_filter, country_filter, exp_filter, paradox_only),
    pn.bind(paradox_top_roles, devtype_filter, country_filter, exp_filter, paradox_only),
)

# -----------------------------------------------------------------------------
# Tab 2, Trust Dynamics
# -----------------------------------------------------------------------------
def trust_by_usage_band(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only).copy()
    f["UsageBand"] = pd.cut(
        f["UsageScore"], bins=[-0.1, 25, 50, 75, 100],
        labels=["Low (0-25)", "Med-Low (25-50)", "Med-High (50-75)", "High (75-100)"],
    )
    s = (f.dropna(subset=["UsageBand"])
            .groupby("UsageBand", observed=True)
            .agg(AvgTrust=("TrustScore", "mean"),
                 AvgFrust=("FrustrationScore", "mean"),
                 N=("ResponseId", "count")).reset_index())
    fig = go.Figure()
    fig.add_bar(x=s["UsageBand"], y=s["AvgTrust"], name="Avg Trust",
                marker_color="#33a02c")
    fig.add_bar(x=s["UsageBand"], y=s["AvgFrust"], name="Avg Frustration",
                marker_color="#ff7f0e")
    fig.update_layout(
        title="Trust and Frustration by Usage Band",
        barmode="group", height=380, yaxis_range=[0, 100],
        yaxis_title="Score (0-100)", margin=dict(t=50, b=40, l=10, r=10),
    )
    for _, r in s.iterrows():
        fig.add_annotation(x=r["UsageBand"], y=-7, text=f"n={int(r['N']):,}",
                           showarrow=False, font_size=10, font_color="#666")
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

def trust_by_experience(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only).copy()
    f["ExpGroup"] = pd.cut(
        f["WorkExpNum"], bins=[-0.1, 1, 5, 10, 20, 60],
        labels=["0-1 yrs", "2-5 yrs", "6-10 yrs", "11-20 yrs", "20+ yrs"],
    )
    s = (f.dropna(subset=["ExpGroup"]).groupby("ExpGroup", observed=True)
          .agg(AvgUsage=("UsageScore", "mean"),
               AvgTrust=("TrustScore", "mean"),
               AvgFrust=("FrustrationScore", "mean"),
               N=("ResponseId", "count")).reset_index())
    fig = go.Figure()
    fig.add_bar(x=s["ExpGroup"], y=s["AvgUsage"], name="Usage",
                marker_color="#2b7cbf")
    fig.add_bar(x=s["ExpGroup"], y=s["AvgTrust"], name="Trust",
                marker_color="#33a02c")
    fig.add_bar(x=s["ExpGroup"], y=s["AvgFrust"], name="Frustration",
                marker_color="#ff7f0e")
    fig.update_layout(
        title="Composite scores by experience bracket",
        barmode="group", height=380, yaxis_range=[0, 100],
        yaxis_title="Score (0-100)", margin=dict(t=50, b=40, l=10, r=10),
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

def trust_by_role(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only).copy()
    s = (f.groupby("DevType")
          .agg(AvgTrust=("TrustScore", "mean"),
               AvgUsage=("UsageScore", "mean"),
               N=("ResponseId", "count")).reset_index())
    s = s[s["N"] >= 50].sort_values("AvgTrust", ascending=True).tail(15)
    if s.empty:
        return pn.pane.Markdown("*Not enough data under the current filters.*")
    fig = px.bar(
        s, y="DevType", x="AvgTrust", orientation="h",
        hover_data={"AvgUsage": ":.1f", "N": True},
        color="AvgUsage", color_continuous_scale="blues",
    )
    fig.update_layout(
        title="Top 15 roles by average trust  (n ≥ 50)",
        xaxis_title="Average Trust Score", yaxis_title=None,
        height=520, margin=dict(t=50, b=40, l=10, r=10),
        xaxis_range=[0, 100],
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

def heatmap_chart(devtypes, countries, exp_range, p_only):
    f = filter_df(quadrants, devtypes, countries, exp_range, p_only).copy()
    f["u_bin"] = (f["UsageScore"] // 10).clip(upper=9).astype(int) * 10
    f["t_bin"] = (f["TrustScore"] // 10).clip(upper=9).astype(int) * 10
    h = (f.groupby(["t_bin", "u_bin"]).size()
          .unstack(fill_value=0)
          .reindex(index=range(0, 100, 10),
                   columns=range(0, 100, 10), fill_value=0))
    fig = px.imshow(
        h.values, x=h.columns, y=h.index, origin="lower",
        color_continuous_scale="viridis", aspect="auto",
        labels=dict(x="Usage Score (binned)", y="Trust Score (binned)",
                    color="Developer Count"),
    )
    fig.update_layout(
        title="Heatmap: Composite Usage × Trust",
        height=440, margin=dict(t=50, b=40, l=10, r=10),
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")

trust_tab = pn.Column(
    pn.bind(trust_by_usage_band, devtype_filter, country_filter, exp_filter, paradox_only),
    pn.bind(trust_by_experience, devtype_filter, country_filter, exp_filter, paradox_only),
    pn.bind(trust_by_role,       devtype_filter, country_filter, exp_filter, paradox_only),
    pn.bind(heatmap_chart,       devtype_filter, country_filter, exp_filter, paradox_only),
)

# -----------------------------------------------------------------------------
# Tab 3, Machine Learning
# -----------------------------------------------------------------------------
# RF feature-importance, fixed (it's about the trained model, not filters)
fi_sorted = rf_imp.sort_values("Importance", ascending=True)
fig_fi = px.bar(
    fi_sorted, x="Importance", y="Feature", orientation="h",
    color="Importance", color_continuous_scale="purples",
)
fig_fi.update_layout(
    title=(f"What predicts the AI Trust Paradox?  "
           f"(Random-Forest ensemble · accuracy {rf_metrics['accuracy']:.1%})"),
    xaxis_title="Mean importance (3-seed RF)", yaxis_title=None,
    height=420, margin=dict(t=60, b=40, l=10, r=10),
)
rf_chart = pn.pane.Plotly(fig_fi, sizing_mode="stretch_width")

# K-Means scan plot, k vs inertia & silhouette
fig_scan = go.Figure()
fig_scan.add_trace(go.Scatter(x=kmeans_scan["k"], y=kmeans_scan["inertia"],
                              mode="lines+markers", name="Inertia (lower = tighter)",
                              yaxis="y", line=dict(color="#1f77b4")))
fig_scan.add_trace(go.Scatter(x=kmeans_scan["k"], y=kmeans_scan["silhouette"],
                              mode="lines+markers", name="Silhouette (higher = cleaner)",
                              yaxis="y2", line=dict(color="#d62728")))
fig_scan.update_layout(
    title="K-Means k-sweep  (run in parallel via Ray)",
    xaxis_title="k (number of clusters)",
    yaxis=dict(title="Inertia", side="left"),
    yaxis2=dict(title="Silhouette", side="right", overlaying="y"),
    height=380, margin=dict(t=50, b=40, l=10, r=10),
)
kmeans_chart = pn.pane.Plotly(fig_scan, sizing_mode="stretch_width")

# Cluster centroids table
cluster_table = pn.widgets.Tabulator(
    cluster_summary.round(2),
    height=240, layout="fit_data_stretch",
    show_index=False,
    name="K-Means cluster centroids (k=4)",
)

# Cluster PCA scatter (using the precomputed cluster labels)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
cluster_features = [
    "UsageScore", "TrustScore", "FrustrationScore",
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    "ProblemCount", "ThreatLevel", "DevEnvToolCount", "WorkExpNum",
]
Xc  = StandardScaler().fit_transform(ray_clusters[cluster_features].values)
pcs = PCA(n_components=2, random_state=0).fit_transform(Xc)
pca_df = ray_clusters.assign(PC1=pcs[:, 0], PC2=pcs[:, 1])
fig_pca = px.scatter(
    pca_df.sample(min(5000, len(pca_df)), random_state=0),
    x="PC1", y="PC2", color="Cluster",
    opacity=0.55,
    color_continuous_scale="viridis",
    hover_data={"UsageScore": ":.1f", "TrustScore": ":.1f",
                "FrustrationScore": ":.1f", "Cluster": True,
                "PC1": False, "PC2": False},
)
fig_pca.update_layout(
    title="Developer clusters in PCA space  (sample of 5,000)",
    height=460, margin=dict(t=50, b=40, l=10, r=10),
)
pca_chart = pn.pane.Plotly(fig_pca, sizing_mode="stretch_width")

ml_explainer = pn.pane.Markdown(
    f"""
### What's running here

**Random Forest**, *predicts whether a developer is in the paradox group*
- 3-seed ensemble (seeds 42 / 1337 / 2025), 200 trees each, run in parallel via Ray
- Predictors: {', '.join(f'`{c}`' for c in rf_metrics['predictors'])}
- **Trust columns are excluded** (they define the label, keeping them would leak)
- Accuracy on held-out 20 % test set: **{rf_metrics['accuracy']:.1%}**
- Train / test sizes: {rf_metrics['n_train']:,} / {rf_metrics['n_test']:,}
- Median split used for the label, Usage = {rf_metrics['median_usage']:.1f}, Trust = {rf_metrics['median_trust']:.1f}

**K-Means**, *unsupervised segmentation of developers*
- k swept 2..6 in parallel via Ray. Final fit at k = 4
- 11 features (the composite scores + their components + experience)
- Doesn't see the paradox label at all, and yet one of the 4 tribes lines up
  with the paradox quadrant, which is independent confirmation that the
  paradox is a real pattern in the data.
"""
)

ml_tab = pn.Column(rf_chart, kmeans_chart, cluster_table, pca_chart, ml_explainer)

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
sidebar = pn.Column(
    pn.pane.Markdown("## Filters"),
    devtype_filter, country_filter, exp_filter, paradox_only,
    pn.pane.Markdown(
        """
---
### About

**AI Trust Paradox**, analysing **49,191** developers from the
2025 Stack Overflow Developer Survey.

**Composite scores (0–100)** built from multiple survey columns:
- **Usage**: AISelect + AIAgents + # AI models + workflow integration
- **Trust**: AIAcc + AIComplex + AISent
- **Frustration**: AIFrustration + AIThreat

Quadrants split on the **median** of the analysed population
(27,258 with full trust data).
"""
    ),
)

tabs = pn.Tabs(
    ("The Paradox",     paradox_tab),
    ("Trust Dynamics",  trust_tab),
    ("Machine Learning", ml_tab),
    dynamic=True,
)

dashboard = pn.template.FastListTemplate(
    title="AI Trust Paradox, Phase 2 Dashboard",
    site="CMP_SC-8540 · Big Data & Model Management",
    sidebar=[sidebar],
    main=[kpi_panel, tabs],
    accent_base_color="#d62728",
    header_background="#1a1a2e",
)
dashboard.servable()

if __name__ == "__main__":
    # `python dashboard/app.py` builds the app objects but doesn't serve.
    # Use `panel serve dashboard/app.py --show` to launch a browser.
    print("\nReady. Run with: panel serve dashboard/app.py --show")
