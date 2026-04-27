"""
AI Trust Paradox Dashboard (Panel + Bokeh)
==========================================

The dashboard only renders views that are backed by available output files.
When regenerated row-level Parquet data is present, an extra filtered profile
tab appears. With the committed summary outputs, it shows native interactive
Bokeh charts, KPI cards, and ML summary tables.

Run:
    panel serve dashboard/app.py --show --autoreload --port 5006
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    FixedTicker,
    HoverTool,
    LabelSet,
    LinearAxis,
    LinearColorMapper,
    NumeralTickFormatter,
    Range1d,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", message=".*allow_refs.*", category=FutureWarning)
pn.extension("tabulator", sizing_mode="stretch_width")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "output"
SPARK_SQL = OUTPUT / "spark_sql_results"
ML = OUTPUT / "ray_ml_results"
CLEANED = OUTPUT / "cleaned_data"

PALETTE = {
    "High Usage - High Trust": "#2f9e44",
    "High Usage - Low Trust": "#c92a2a",
    "Low Usage - High Trust": "#2b7cbf",
    "Low Usage - Low Trust": "#6c757d",
}

ROLE_LABELS = {
    "Developer, AI apps or physical AI": "AI app developer",
    "AI/ML engineer": "AI/ML engineer",
    "Developer, embedded applications or devices": "Embedded developer",
    "Developer, game or graphics": "Game/graphics developer",
}

EXPERIENCE_ORDER = [
    "Beginner: 0-1 years",
    "Junior: 2-5 years",
    "Mid-level: 6-10 years",
    "Senior: 11-20 years",
    "Expert: 20+ years",
    "Unknown",
]

EXPERIENCE_SHORT = {
    "Beginner: 0-1 years": "0-1 yrs",
    "Junior: 2-5 years": "2-5 yrs",
    "Mid-level: 6-10 years": "6-10 yrs",
    "Senior: 11-20 years": "11-20 yrs",
    "Expert: 20+ years": "20+ yrs",
    "Unknown": "Unknown",
}

USAGE_ORDER = {
    "Yes, I use AI tools daily": 0,
    "Yes, I use AI tools weekly": 1,
    "Yes, I use AI tools monthly or infrequently": 2,
    "No, but I plan to soon": 3,
    "No, and I don't plan to": 4,
    "Unknown": 5,
}

USAGE_LABELS = {
    "Yes, I use AI tools daily": "Daily",
    "Yes, I use AI tools weekly": "Weekly",
    "Yes, I use AI tools monthly or infrequently": "Monthly / infrequent",
    "No, but I plan to soon": "Plan to soon",
    "No, and I don't plan to": "Do not plan to",
    "Unknown": "Unknown",
}

FEATURE_LABELS = {
    "WorkExpNum": "Work experience",
    "WorkflowIntegration": "Workflow integration",
    "AIModelCount": "AI model count",
    "DevEnvToolCount": "Developer-environment tools",
    "AgentDepth": "Agent usage depth",
    "UsageFreq": "AI usage frequency",
    "ProblemCount": "AI problem count",
    "ThreatLevel": "AI threat perception",
}

CLUSTER_PROFILES = {
    0: {
        "Segment": "Power users",
        "Meaning": "High usage and high trust, with broad model/tool use and lower frustration.",
    },
    1: {
        "Segment": "Frustrated adopters",
        "Meaning": "Active AI users with the highest frustration and threat concern.",
    },
    2: {
        "Segment": "Regular low-friction users",
        "Meaning": "The largest segment: moderate trust, regular use, and low frustration.",
    },
    3: {
        "Segment": "Low-adoption skeptics",
        "Meaning": "Low usage and the lowest trust; AI is not deeply integrated into work.",
    },
}


def spark_csv(name):
    files = sorted((SPARK_SQL / name).glob("part-*.csv"))
    if not files:
        raise FileNotFoundError(f"Missing output/spark_sql_results/{name}/part-*.csv")
    return pd.read_csv(files[0])


def as_number(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def friendly_role(role):
    return ROLE_LABELS.get(str(role), str(role).replace("Developer, ", "").replace(" or ", " / "))


def wrapped_label(value, width=24):
    words = str(value).split()
    lines = []
    line = []
    for word in words:
        if line and sum(len(part) for part in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)


def note(text):
    return pn.pane.HTML(f'<div class="data-note">{text}</div>', sizing_mode="stretch_width")


BOKEH_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def bokeh_card(plot, caption):
    return pn.Column(
        pn.pane.Bokeh(plot, sizing_mode="stretch_width"),
        pn.pane.HTML(f'<div class="chart-caption">{caption}</div>', sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        css_classes=["chart-card"],
    )


def style_bokeh(plot):
    plot.toolbar.autohide = True
    plot.outline_line_color = None
    plot.border_fill_color = "white"
    plot.background_fill_color = "white"
    plot.grid.grid_line_color = "#edf1f5"
    plot.axis.axis_label_text_color = "#1f2933"
    plot.axis.major_label_text_color = "#1f2933"
    plot.title.text_color = "#111827"
    plot.title.text_font_size = "16px"
    plot.title.text_font_style = "bold"
    return plot


def interactive_quadrant_chart(data=None, title="Trust-usage quadrant sizes"):
    data = quadrant_summary.copy() if data is None else data.copy()
    if data.empty:
        return note("No quadrant rows are available for this view.")

    order = [
        "Low Usage - Low Trust",
        "Low Usage - High Trust",
        "High Usage - Low Trust",
        "High Usage - High Trust",
    ]
    data["Order"] = data["TrustUsageGroup"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("Order")
    data["Label"] = data["TrustUsageGroup"]
    data["ShareLabel"] = (data["DeveloperCount"] / data["DeveloperCount"].sum() * 100).map(lambda v: f"{v:.1f}%")
    data["CountLabel"] = data["DeveloperCount"].map(lambda v: f"{int(v):,}")
    data["Color"] = data["TrustUsageGroup"].map(PALETTE)
    source = ColumnDataSource(data)

    plot = figure(
        y_range=data["Label"].tolist(),
        height=430,
        sizing_mode="stretch_width",
        title=title,
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Developers",
    )
    bars = plot.hbar(y="Label", right="DeveloperCount", height=0.58, color="Color", source=source)
    plot.add_layout(
        LabelSet(
            x="DeveloperCount",
            y="Label",
            text="CountLabel",
            source=source,
            x_offset=8,
            y_offset=-7,
            text_font_size="11px",
            text_color="#344054",
        )
    )
    plot.x_range.start = 0
    plot.x_range.end = float(data["DeveloperCount"].max()) * 1.18
    plot.xaxis.formatter = NumeralTickFormatter(format="0,0")
    plot.ygrid.grid_line_color = None
    plot.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("Group", "@TrustUsageGroup"),
                ("Developers", "@CountLabel"),
                ("Share", "@ShareLabel"),
                ("Avg usage", "@AvgUsageScore{0.0}"),
                ("Avg trust", "@AvgTrustScore{0.0}"),
                ("Avg frustration", "@AvgFrustrationScore{0.0}"),
            ],
        )
    )
    return bokeh_card(
        style_bokeh(plot),
        "Median-split trust/usage groups. The red bar is the high-usage, low-trust paradox group.",
    )


def interactive_trust_usage_chart():
    data = trust_by_usage.copy()
    order = ["Low (0-25)", "Medium-Low (25-50)", "Medium-High (50-75)", "High (75-100)"]
    data["Order"] = data["UsageBand"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("Order").reset_index(drop=True)
    data["x"] = data.index
    data["TrustLabel"] = data["AvgTrustScore"].map(lambda v: f"{v:.1f}")
    data["FrustrationLabel"] = data["AvgFrustrationScore"].map(lambda v: f"{v:.1f}")
    data["CountLabel"] = data["DeveloperCount"].map(lambda v: f"{int(v):,}")
    source = ColumnDataSource(data)

    plot = figure(
        height=430,
        sizing_mode="stretch_width",
        title="Trust and frustration by AI usage band",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="AI usage band",
        y_axis_label="Average score",
        y_range=(0, 82),
    )
    trust_line = plot.line("x", "AvgTrustScore", source=source, color="#2f9e44", line_width=3, legend_label="Trust")
    trust_points = plot.scatter("x", "AvgTrustScore", source=source, color="#2f9e44", size=9, legend_label="Trust")
    frust_line = plot.line("x", "AvgFrustrationScore", source=source, color="#f08c00", line_width=3, legend_label="Frustration")
    frust_points = plot.scatter("x", "AvgFrustrationScore", source=source, color="#f08c00", size=9, legend_label="Frustration")
    plot.add_layout(LabelSet(x="x", y="AvgTrustScore", text="TrustLabel", source=source, y_offset=9, text_font_size="11px"))
    plot.add_layout(LabelSet(x="x", y="AvgFrustrationScore", text="FrustrationLabel", source=source, y_offset=9, text_font_size="11px"))
    plot.xaxis.ticker = FixedTicker(ticks=data["x"].tolist())
    plot.xaxis.major_label_overrides = {int(row.x): row.UsageBand for row in data.itertuples()}
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.add_tools(
        HoverTool(
            renderers=[trust_points, frust_points],
            tooltips=[
                ("Usage band", "@UsageBand"),
                ("Developers", "@CountLabel"),
                ("Trust", "@AvgTrustScore{0.0}"),
                ("Frustration", "@AvgFrustrationScore{0.0}"),
            ],
        )
    )
    return bokeh_card(
        style_bokeh(plot),
        "Trust rises with heavier AI usage, but frustration also rises.",
    )


def interactive_experience_chart():
    data = experience.copy().reset_index(drop=True)
    data["Label"] = data["ExperienceGroup"].map(EXPERIENCE_SHORT)
    data["x"] = data.index
    data["TrustLabel"] = data["AvgTrustScore"].map(lambda v: f"{v:.1f}")
    data["CountLabel"] = data["DeveloperCount"].map(lambda v: f"{int(v):,}")
    source = ColumnDataSource(data)
    ymin = float(data["AvgTrustScore"].min()) - 0.8
    ymax = float(data["AvgTrustScore"].max()) + 0.8

    plot = figure(
        height=410,
        sizing_mode="stretch_width",
        title="Experience barely changes AI trust (zoomed scale)",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Experience group",
        y_axis_label="Average trust score",
        y_range=(ymin, ymax),
    )
    line = plot.line("x", "AvgTrustScore", source=source, color="#2f9e44", line_width=3)
    points = plot.scatter("x", "AvgTrustScore", source=source, color="#2f9e44", size=9)
    plot.add_layout(LabelSet(x="x", y="AvgTrustScore", text="TrustLabel", source=source, y_offset=9, text_font_size="11px"))
    plot.xaxis.ticker = FixedTicker(ticks=data["x"].tolist())
    plot.xaxis.major_label_overrides = {int(row.x): row.Label for row in data.itertuples()}
    plot.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                ("Experience", "@ExperienceGroup"),
                ("Developers", "@CountLabel"),
                ("Avg trust", "@AvgTrustScore{0.00}"),
                ("Avg usage", "@AvgUsageScore{0.0}"),
            ],
        )
    )
    return bokeh_card(
        style_bokeh(plot),
        f"Known experience groups vary by only {experience_trust_range:.2f} trust points.",
    )


def interactive_usage_distribution_chart():
    data = ai_usage.copy()
    data["Order"] = data["AISelect"].map(USAGE_ORDER).fillna(99)
    data["Label"] = data["AISelect"].map(USAGE_LABELS).fillna(data["AISelect"])
    data = data.sort_values("Order", ascending=False)
    data["CountLabel"] = data["DeveloperCount"].map(lambda v: f"{int(v):,}")
    data["Color"] = np.where(data["Label"] == "Unknown", "#6c757d", "#2b7cbf")
    source = ColumnDataSource(data)

    plot = figure(
        y_range=data["Label"].tolist(),
        height=410,
        sizing_mode="stretch_width",
        title="AI usage frequency in the cleaned survey",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Developers",
    )
    bars = plot.hbar(y="Label", right="DeveloperCount", height=0.58, color="Color", source=source)
    plot.add_layout(LabelSet(x="DeveloperCount", y="Label", text="CountLabel", source=source, x_offset=8, y_offset=-7, text_font_size="11px"))
    plot.x_range.start = 0
    plot.x_range.end = float(data["DeveloperCount"].max()) * 1.18
    plot.xaxis.formatter = NumeralTickFormatter(format="0,0")
    plot.ygrid.grid_line_color = None
    plot.add_tools(HoverTool(renderers=[bars], tooltips=[("Usage", "@AISelect"), ("Developers", "@CountLabel")]))
    return bokeh_card(style_bokeh(plot), "Raw AI usage-frequency distribution from the cleaned survey.")


def interactive_role_trust_chart():
    data = trust_by_role[trust_by_role["DeveloperCount"] >= 100].copy()
    data = pd.concat([data.nsmallest(5, "AvgTrustScore"), data.nlargest(8, "AvgTrustScore")]).drop_duplicates("DevType")
    data = data.sort_values("AvgTrustScore")
    data["RoleLabel"] = data["DevType"].map(friendly_role)
    data["TrustLabel"] = data["AvgTrustScore"].map(lambda v: f"{v:.1f}")
    data["CountLabel"] = data["DeveloperCount"].map(lambda v: f"{int(v):,}")
    data["Color"] = np.where(data["AvgTrustScore"] >= data["AvgTrustScore"].median(), "#2f9e44", "#c92a2a")
    source = ColumnDataSource(data)

    plot = figure(
        y_range=data["RoleLabel"].tolist(),
        height=570,
        sizing_mode="stretch_width",
        title="Roles with the highest and lowest AI trust",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Average trust score",
    )
    bars = plot.hbar(y="RoleLabel", right="AvgTrustScore", height=0.58, color="Color", source=source)
    plot.add_layout(LabelSet(x="AvgTrustScore", y="RoleLabel", text="TrustLabel", source=source, x_offset=8, y_offset=-7, text_font_size="11px"))
    plot.x_range.start = max(0, float(data["AvgTrustScore"].min()) - 3)
    plot.x_range.end = min(100, float(data["AvgTrustScore"].max()) + 5)
    plot.ygrid.grid_line_color = None
    plot.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("Role", "@DevType"),
                ("Developers", "@CountLabel"),
                ("Avg trust", "@AvgTrustScore{0.0}"),
                ("Avg usage", "@AvgUsageScore{0.0}"),
                ("Avg frustration", "@AvgFrustrationScore{0.0}"),
            ],
        )
    )
    return bokeh_card(style_bokeh(plot), "Role explains trust differences much more clearly than experience.")


def interactive_heatmap_chart():
    data = ray_clusters[["UsageScore", "TrustScore"]].dropna().copy()
    data["u_bin"] = (data["UsageScore"] // 10).clip(upper=9).astype(int) * 10
    data["t_bin"] = (data["TrustScore"] // 10).clip(upper=9).astype(int) * 10
    heat = data.groupby(["u_bin", "t_bin"]).size().reset_index(name="Developers")
    heat["x"] = heat["u_bin"] + 5
    heat["y"] = heat["t_bin"] + 5
    heat["UsageBin"] = heat["u_bin"].map(lambda v: f"{v}-{v + 10}")
    heat["TrustBin"] = heat["t_bin"].map(lambda v: f"{v}-{v + 10}")
    mapper = LinearColorMapper(palette=Viridis256, low=0, high=float(heat["Developers"].max()))
    source = ColumnDataSource(heat)

    plot = figure(
        height=500,
        sizing_mode="stretch_width",
        title="Where developers cluster by usage and trust",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Usage score",
        y_axis_label="Trust score",
        x_range=(0, 100),
        y_range=(0, 100),
    )
    rects = plot.rect(x="x", y="y", width=10, height=10, source=source, fill_color={"field": "Developers", "transform": mapper}, line_color=None)
    plot.add_layout(ColorBar(color_mapper=mapper, ticker=BasicTicker(), label_standoff=8, title="Developers"), "right")
    plot.add_tools(HoverTool(renderers=[rects], tooltips=[("Usage score", "@UsageBin"), ("Trust score", "@TrustBin"), ("Developers", "@Developers{0,0}")]))
    return bokeh_card(style_bokeh(plot), "Density view of composite usage and trust scores.")


def interactive_feature_importance_chart():
    data = rf_imp.sort_values("Importance").copy()
    data["FeatureLabel"] = data["Feature"].map(FEATURE_LABELS).fillna(data["Feature"])
    data["ImportancePct"] = data["Importance"] * 100
    data["ImportanceLabel"] = data["ImportancePct"].map(lambda v: f"{v:.1f}%")
    source = ColumnDataSource(data)

    plot = figure(
        y_range=data["FeatureLabel"].tolist(),
        height=430,
        sizing_mode="stretch_width",
        title="Random Forest drivers of high-usage, low-trust membership",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Mean feature importance",
    )
    bars = plot.hbar(y="FeatureLabel", right="ImportancePct", height=0.58, color="#2b7cbf", source=source)
    plot.add_layout(LabelSet(x="ImportancePct", y="FeatureLabel", text="ImportanceLabel", source=source, x_offset=8, y_offset=-7, text_font_size="11px"))
    plot.x_range.start = 0
    plot.x_range.end = float(data["ImportancePct"].max()) * 1.22
    plot.ygrid.grid_line_color = None
    plot.add_tools(HoverTool(renderers=[bars], tooltips=[("Feature", "@Feature"), ("Importance", "@ImportanceLabel")]))
    return bokeh_card(style_bokeh(plot), "Trust fields are excluded, so this does not leak the label definition.")


def interactive_kmeans_scan_panel():
    data = kmeans_scan.copy()
    data["InertiaLabel"] = data["inertia"].map(lambda v: f"{v:,.0f}")
    data["SilhouetteLabel"] = data["silhouette"].map(lambda v: f"{v:.3f}")
    source = ColumnDataSource(data)

    inertia = figure(
        height=320,
        sizing_mode="stretch_width",
        title="K-Means inertia by k",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Clusters (k)",
        y_axis_label="Inertia",
    )
    inertia_line = inertia.line("k", "inertia", source=source, color="#2b7cbf", line_width=3)
    inertia_points = inertia.scatter("k", "inertia", source=source, color="#2b7cbf", size=9)
    inertia.yaxis.formatter = NumeralTickFormatter(format="0,0")
    inertia.add_tools(HoverTool(renderers=[inertia_points], tooltips=[("k", "@k"), ("Inertia", "@InertiaLabel")]))

    silhouette = figure(
        height=320,
        sizing_mode="stretch_width",
        title="K-Means silhouette by k",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Clusters (k)",
        y_axis_label="Silhouette",
    )
    silhouette_line = silhouette.line("k", "silhouette", source=source, color="#c92a2a", line_width=3)
    silhouette_points = silhouette.scatter("k", "silhouette", source=source, color="#c92a2a", size=9)
    silhouette.add_tools(HoverTool(renderers=[silhouette_points], tooltips=[("k", "@k"), ("Silhouette", "@SilhouetteLabel")]))

    return pn.Column(
        pn.Row(bokeh_card(style_bokeh(inertia), "Lower inertia is better, but it always falls as k increases."),
               bokeh_card(style_bokeh(silhouette), "Silhouette is highest at k=2; k=4 is used for interpretability.")),
        sizing_mode="stretch_width",
    )


def interactive_cluster_pca_chart():
    cluster_features = [
        "UsageScore",
        "TrustScore",
        "FrustrationScore",
        "UsageFreq",
        "AgentDepth",
        "AIModelCount",
        "WorkflowIntegration",
        "ProblemCount",
        "ThreatLevel",
        "DevEnvToolCount",
        "WorkExpNum",
    ]
    sample = ray_clusters.dropna(subset=cluster_features + ["Cluster"]).copy()
    if len(sample) > 7000:
        sample = sample.sample(7000, random_state=0)
    pcs = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(sample[cluster_features].values))
    colors = {0: "#2b7cbf", 1: "#f08c00", 2: "#2f9e44", 3: "#c92a2a"}
    sample = sample.assign(
        PC1=pcs[:, 0],
        PC2=pcs[:, 1],
        ClusterInt=sample["Cluster"].astype(int),
        ClusterLabel=sample["Cluster"].map(lambda value: f"Cluster {int(value)}: {CLUSTER_PROFILES[int(value)]['Segment']}"),
        Color=sample["Cluster"].map(lambda value: colors[int(value)]),
    )
    source = ColumnDataSource(sample)

    plot = figure(
        height=520,
        sizing_mode="stretch_width",
        title="Ray K-Means developer segments projected into PCA space",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="PC 1",
        y_axis_label="PC 2",
    )
    points = plot.scatter("PC1", "PC2", source=source, color="Color", alpha=0.5, size=5)
    plot.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                ("Cluster", "@ClusterLabel"),
                ("Usage", "@UsageScore{0.0}"),
                ("Trust", "@TrustScore{0.0}"),
                ("Frustration", "@FrustrationScore{0.0}"),
            ],
        )
    )
    return bokeh_card(
        style_bokeh(plot),
        "Each point is a developer. PCA compresses 11 standardized features into two viewable dimensions.",
    )


def kpi_card(label, value, sublabel, tone="#2b7cbf"):
    return pn.pane.HTML(
        f"""
        <div class="kpi-card" style="border-top-color:{tone};">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sublabel}</div>
        </div>
        """,
        sizing_mode="stretch_width",
    )


def compact_metric_card(label, value, sublabel, tone):
    return f"""
    <div class="metric-card" style="border-top-color:{tone};">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sublabel}</div>
    </div>
    """


print("loading dashboard outputs ...")
dataset_summary = spark_csv("dataset_summary")
ai_usage = spark_csv("ai_usage_distribution")
trust_by_usage = spark_csv("trust_by_usage")
trust_by_role = spark_csv("trust_by_role")
experience = spark_csv("experience_analysis")
quadrant_summary = spark_csv("quadrant_summary")

cluster_summary = pd.read_csv(ML / "cluster_summary.csv")
ray_clusters = pd.read_csv(ML / "ray_cluster_results.csv")
rf_imp = pd.read_csv(ML / "paradox_feature_importance.csv")
rf_predictions = pd.read_csv(ML / "rf_predictions.csv")
kmeans_scan = pd.read_csv(ML / "kmeans_scan.csv")
rf_metrics = json.loads((ML / "rf_metrics.json").read_text())

try:
    row_quadrants = pd.read_parquet(CLEANED / "ai_trust_quadrants")
    HAS_ROW_DATA = True
except Exception as exc:
    row_quadrants = None
    HAS_ROW_DATA = False
    print(f"row-level parquet unavailable; using summary outputs only: {exc}")

total_responses = int(dataset_summary.loc[0, "TotalResponses"])
respondents_with_trust = int(dataset_summary.loc[0, "RespondentsWithTrust"])
country_count = int(dataset_summary.loc[0, "CountryCount"])
role_count = int(dataset_summary.loc[0, "DeveloperRoleCount"])
avg_usage = as_number(dataset_summary.loc[0, "AvgUsageScore"])
avg_trust = as_number(dataset_summary.loc[0, "AvgTrustScore"])
avg_frustration = as_number(dataset_summary.loc[0, "AvgFrustrationScore"])

paradox_count = int(
    quadrant_summary.loc[
        quadrant_summary["TrustUsageGroup"] == "High Usage - Low Trust", "DeveloperCount"
    ].iloc[0]
)
paradox_share = paradox_count / respondents_with_trust * 100

experience = (
    experience.assign(_order=experience["ExperienceGroup"].map({v: i for i, v in enumerate(EXPERIENCE_ORDER)}))
    .sort_values("_order")
    .drop(columns="_order")
)
known_experience = experience[experience["ExperienceGroup"] != "Unknown"]
experience_trust_range = (
    known_experience["AvgTrustScore"].max() - known_experience["AvgTrustScore"].min()
)

highest_role = trust_by_role.sort_values("AvgTrustScore", ascending=False).iloc[0]
lowest_role = trust_by_role.sort_values("AvgTrustScore", ascending=True).iloc[0]

cluster_summary_display = cluster_summary.copy()
cluster_summary_display["Segment"] = cluster_summary_display["Cluster"].map(
    lambda c: CLUSTER_PROFILES[int(c)]["Segment"]
)
cluster_summary_display["Meaning"] = cluster_summary_display["Cluster"].map(
    lambda c: CLUSTER_PROFILES[int(c)]["Meaning"]
)
cluster_summary_display = cluster_summary_display[
    [
        "Cluster",
        "Segment",
        "Meaning",
        "DeveloperCount",
        "UsageScore",
        "TrustScore",
        "FrustrationScore",
        "AgentDepth",
        "AIModelCount",
        "WorkflowIntegration",
        "WorkExpNum",
    ]
].sort_values("Cluster")

model_eval = {
    "accuracy": as_number(rf_metrics["accuracy"]),
    "baseline_accuracy": as_number((rf_predictions["y_true"] == 0).mean()),
    "balanced_accuracy": balanced_accuracy_score(rf_predictions["y_true"], rf_predictions["y_pred"]),
    "precision": precision_score(rf_predictions["y_true"], rf_predictions["y_pred"], zero_division=0),
    "recall": recall_score(rf_predictions["y_true"], rf_predictions["y_pred"], zero_division=0),
    "f1": f1_score(rf_predictions["y_true"], rf_predictions["y_pred"], zero_division=0),
    "roc_auc": roc_auc_score(rf_predictions["y_true"], rf_predictions["y_proba"]),
}

print(f"  total rows: {total_responses:,}")
print(f"  analysed trust rows: {respondents_with_trust:,}")
print(f"  dashboard mode: {'row-level profile enabled' if HAS_ROW_DATA else 'summary outputs only'}")


def header_panel():
    mode_text = "Row-level profile enabled" if HAS_ROW_DATA else "Summary outputs only"
    scope_text = (
        "Country and filtered role charts are available because row-level Parquet was found."
        if HAS_ROW_DATA
        else "Country charts are hidden because the current committed outputs do not include row-level country data."
    )
    return pn.pane.HTML(
        f"""
        <section class="insight-strip">
          <div>
            <div class="eyebrow">AI Trust Paradox</div>
            <h2>Usage and role explain trust more clearly than years of experience.</h2>
            <p>
              AI app developers average <b>{highest_role['AvgTrustScore']:.1f}</b> trust,
              while {friendly_role(lowest_role['DevType']).lower()} average
              <b>{lowest_role['AvgTrustScore']:.1f}</b>. Across known experience groups,
              trust moves only <b>{experience_trust_range:.1f}</b> points.
            </p>
          </div>
          <div>
            <span class="mode-badge">{mode_text}</span>
            <div class="scope-note">{scope_text}</div>
          </div>
        </section>
        """,
        sizing_mode="stretch_width",
    )


def kpi_panel():
    return pn.pane.HTML(
        f"""
        <div class="metric-strip">
          {compact_metric_card("Analysed", f"{respondents_with_trust:,}", "trust rows", "#2f9e44")}
          {compact_metric_card("Paradox", f"{paradox_count:,}", f"{paradox_share:.1f}% of analysed", "#c92a2a")}
          {compact_metric_card("Usage", f"{avg_usage:.1f}", "avg score", "#2b7cbf")}
          {compact_metric_card("Trust", f"{avg_trust:.1f}", "avg score", "#2f9e44")}
          {compact_metric_card("Frustration", f"{avg_frustration:.1f}", "avg score", "#f08c00")}
          {compact_metric_card("Roles", f"{role_count:,}", f"{country_count:,} countries", "#7950f2")}
        </div>
        """,
        sizing_mode="stretch_width",
    )


def model_eval_cards():
    return pn.pane.HTML(
        f"""
        <div class="metric-strip metric-strip-ml">
          {compact_metric_card("ROC-AUC", f"{model_eval['roc_auc']:.3f}", "ranking", "#2b7cbf")}
          {compact_metric_card("Balanced acc.", f"{model_eval['balanced_accuracy']:.3f}", "class-balanced", "#7950f2")}
          {compact_metric_card("Recall", f"{model_eval['recall']:.1%}", "paradox caught", "#c92a2a")}
          {compact_metric_card("Precision", f"{model_eval['precision']:.1%}", "positive accuracy", "#f08c00")}
          {compact_metric_card("Accuracy", f"{model_eval['accuracy']:.1%}", f"baseline {model_eval['baseline_accuracy']:.1%}", "#6c757d")}
        </div>
        """,
        sizing_mode="stretch_width",
    )


def cluster_cards():
    cards = []
    for _, row in cluster_summary_display.iterrows():
        cluster = int(row["Cluster"])
        cards.append(
            pn.pane.HTML(
                f"""
                <div class="cluster-card">
                  <div class="cluster-title">Cluster {cluster}: {row['Segment']}</div>
                  <p>{row['Meaning']}</p>
                  <div class="cluster-stats">
                    <span>{int(row['DeveloperCount']):,} developers</span>
                    <span>Usage {row['UsageScore']:.1f}</span>
                    <span>Trust {row['TrustScore']:.1f}</span>
                    <span>Frustration {row['FrustrationScore']:.1f}</span>
                  </div>
                </div>
                """,
                sizing_mode="stretch_width",
            )
        )
    return pn.GridBox(*cards, ncols=2, sizing_mode="stretch_width")


def cluster_table():
    return pn.widgets.Tabulator(
        cluster_summary_display.round(2),
        height=260,
        layout="fit_data_stretch",
        show_index=False,
        disabled=True,
    )


def kmeans_table():
    data = kmeans_scan.copy()
    data["inertia"] = data["inertia"].map(lambda value: f"{value:,.0f}")
    data["silhouette"] = data["silhouette"].map(lambda value: f"{value:.3f}")
    data = data.rename(columns={"k": "Clusters (k)", "inertia": "Inertia", "silhouette": "Silhouette"})
    return pn.Column(
        pn.widgets.Tabulator(
            data,
            height=210,
            layout="fit_data_stretch",
            show_index=False,
            disabled=True,
        ),
        pn.pane.HTML(
            """
            <div class="data-note">
              The sweep table is clearer than a dual-axis chart here. Silhouette is highest at k=2,
              while k=4 is used for an interpretable four-segment story aligned with the
              trust-usage analysis.
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def model_notes():
    return pn.pane.HTML(
        f"""
        <div class="model-note">
          <b>What the clusters mean:</b> Ray K-Means does not assign job titles or countries.
          It groups developers with similar numeric behavior across usage, trust, frustration,
          agent depth, model count, workflow integration, problem count, threat concern,
          developer-environment breadth, and work experience. The PCA chart is a 2D projection
          of those 11 standardized features, so PC 1 and PC 2 are compressed feature mixtures.
          The Random Forest model predicts membership in the high-usage, low-trust group
          without using trust fields. Because only about <b>{rf_metrics['positive_rate_test']:.1%}</b>
          of the test set is in that group, plain accuracy is not enough by itself:
          the always-negative baseline is <b>{model_eval['baseline_accuracy']:.1%}</b>,
          while the model accuracy is <b>{model_eval['accuracy']:.1%}</b>.
          ROC-AUC is <b>{model_eval['roc_auc']:.3f}</b>, so the probabilities rank developers
          reasonably well, but the default threshold catches only <b>{model_eval['recall']:.1%}</b>
          of paradox cases. Train/test rows:
          <b>{rf_metrics['n_train']:,}</b>/<b>{rf_metrics['n_test']:,}</b>.
        </div>
        """,
        sizing_mode="stretch_width",
    )


def about_panel():
    filter_text = (
        "Role, country, and work-experience filters are active because row-level Parquet exists."
        if HAS_ROW_DATA
        else "Role/country filters need regenerated row-level Parquet. The current dashboard uses summary Spark SQL and Ray outputs."
    )
    return pn.pane.HTML(
        f"""
        <div class="about-panel">
          <h3>About</h3>
          <p>
            <b>AI Trust Paradox</b> analyses <b>{total_responses:,}</b> developers from the
            2025 Stack Overflow Developer Survey.
          </p>
          <p>
            <b>Composite scores (0-100)</b> combine multiple survey columns:
          </p>
          <ul>
            <li><b>Usage:</b> AISelect + AIAgents + model breadth + workflow integration</li>
            <li><b>Trust:</b> AI accuracy + complex-task ability + sentiment</li>
            <li><b>Frustration:</b> AI problems + AI threat perception</li>
          </ul>
          <p>
            Quadrants split on median usage <b>{rf_metrics['median_usage']:.1f}</b>
            and median trust <b>{rf_metrics['median_trust']:.1f}</b> across
            <b>{respondents_with_trust:,}</b> analysed developers.
          </p>
          <p class="small-note">{filter_text}</p>
        </div>
        """,
        sizing_mode="stretch_width",
    )


def filtered_row_data(devtypes, countries, exp_range, include_unknown_exp):
    data = row_quadrants.copy()
    if devtypes:
        data = data[data["DevType"].isin(devtypes)]
    if countries:
        data = data[data["Country"].isin(countries)]
    if "WorkExpNum" in data.columns:
        lo, hi = exp_range
        exp_mask = data["WorkExpNum"].between(lo, hi)
        if include_unknown_exp:
            exp_mask = exp_mask | data["WorkExpNum"].isna()
        data = data[exp_mask]
    return data


def filtered_kpis(devtypes, countries, exp_range, include_unknown_exp):
    data = filtered_row_data(devtypes, countries, exp_range, include_unknown_exp)
    n = len(data)
    if n == 0:
        return note("No rows match the current filters.")
    paradox_n = int((data["TrustUsageGroup"] == "High Usage - Low Trust").sum())
    return pn.FlexBox(
        kpi_card("Filtered developers", f"{n:,}", "Rows matching the filters"),
        kpi_card("High-usage, low-trust", f"{paradox_n:,}", f"{paradox_n / n * 100:.1f}% of filtered rows", "#c92a2a"),
        kpi_card("Usage", f"{data['UsageScore'].mean():.1f}", "Filtered average", "#2b7cbf"),
        kpi_card("Trust", f"{data['TrustScore'].mean():.1f}", "Filtered average", "#2f9e44"),
        kpi_card("Frustration", f"{data['FrustrationScore'].mean():.1f}", "Filtered average", "#f08c00"),
        sizing_mode="stretch_width",
    )


def filtered_quadrant_chart(devtypes, countries, exp_range, include_unknown_exp):
    data = filtered_row_data(devtypes, countries, exp_range, include_unknown_exp)
    if data.empty:
        return note("No rows match the current filters.")
    grouped = (
        data.groupby("TrustUsageGroup", as_index=False)
        .agg(
            DeveloperCount=("TrustUsageGroup", "size"),
            AvgUsageScore=("UsageScore", "mean"),
            AvgTrustScore=("TrustScore", "mean"),
            AvgFrustrationScore=("FrustrationScore", "mean"),
        )
    )
    return interactive_quadrant_chart(grouped, "Filtered trust-usage quadrants")


def filtered_country_chart(devtypes, countries, exp_range, include_unknown_exp):
    data = filtered_row_data(devtypes, countries, exp_range, include_unknown_exp)
    data = data[data["TrustUsageGroup"] == "High Usage - Low Trust"]
    if data.empty:
        return note("No high-usage, low-trust rows match the current filters.")

    grouped = (
        data.groupby("Country", as_index=False)
        .agg(Developers=("Country", "size"), AvgFrustration=("FrustrationScore", "mean"))
        .sort_values("Developers", ascending=False)
        .head(12)
        .sort_values("Developers")
    )
    grouped["CountLabel"] = grouped["Developers"].map(lambda value: f"{int(value):,}")
    source = ColumnDataSource(grouped)
    plot = figure(
        y_range=grouped["Country"].tolist(),
        height=470,
        sizing_mode="stretch_width",
        title="Countries in the filtered high-usage, low-trust group",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Developers",
    )
    bars = plot.hbar(y="Country", right="Developers", height=0.58, color="#f08c00", source=source)
    plot.add_layout(LabelSet(x="Developers", y="Country", text="CountLabel", source=source, x_offset=8, y_offset=-7, text_font_size="11px"))
    plot.x_range.start = 0
    plot.x_range.end = float(grouped["Developers"].max()) * 1.18
    plot.ygrid.grid_line_color = None
    plot.add_tools(HoverTool(renderers=[bars], tooltips=[("Country", "@Country"), ("Developers", "@CountLabel"), ("Avg frustration", "@AvgFrustration{0.0}")]))
    return bokeh_card(style_bokeh(plot), "Filtered to the high-usage, low-trust group.")


def filtered_role_chart(devtypes, countries, exp_range, include_unknown_exp):
    data = filtered_row_data(devtypes, countries, exp_range, include_unknown_exp)
    data = data[data["TrustUsageGroup"] == "High Usage - Low Trust"]
    if data.empty:
        return note("No high-usage, low-trust rows match the current filters.")

    grouped = (
        data.groupby("DevType", as_index=False)
        .agg(Developers=("DevType", "size"), AvgFrustration=("FrustrationScore", "mean"))
        .sort_values("Developers", ascending=False)
        .head(12)
        .sort_values("Developers")
    )
    grouped["RoleLabel"] = grouped["DevType"].map(friendly_role)
    grouped["CountLabel"] = grouped["Developers"].map(lambda value: f"{int(value):,}")
    source = ColumnDataSource(grouped)
    plot = figure(
        y_range=grouped["RoleLabel"].tolist(),
        height=520,
        sizing_mode="stretch_width",
        title="Roles in the filtered high-usage, low-trust group",
        tools=BOKEH_TOOLS,
        toolbar_location="above",
        x_axis_label="Developers",
    )
    bars = plot.hbar(y="RoleLabel", right="Developers", height=0.58, color="#f08c00", source=source)
    plot.add_layout(LabelSet(x="Developers", y="RoleLabel", text="CountLabel", source=source, x_offset=8, y_offset=-7, text_font_size="11px"))
    plot.x_range.start = 0
    plot.x_range.end = float(grouped["Developers"].max()) * 1.18
    plot.ygrid.grid_line_color = None
    plot.add_tools(HoverTool(renderers=[bars], tooltips=[("Role", "@DevType"), ("Developers", "@CountLabel"), ("Avg frustration", "@AvgFrustration{0.0}")]))
    return bokeh_card(style_bokeh(plot), "Filtered to the high-usage, low-trust group.")


overview_tab = pn.Column(
    pn.Row(interactive_quadrant_chart(), interactive_trust_usage_chart()),
    interactive_heatmap_chart(),
    sizing_mode="stretch_width",
)

trust_tab = pn.Column(
    pn.Row(interactive_experience_chart(), interactive_usage_distribution_chart()),
    interactive_role_trust_chart(),
    sizing_mode="stretch_width",
)

ml_tab = pn.Column(
    cluster_cards(),
    interactive_cluster_pca_chart(),
    model_eval_cards(),
    interactive_feature_importance_chart(),
    interactive_kmeans_scan_panel(),
    cluster_table(),
    model_notes(),
    sizing_mode="stretch_width",
)

tabs_content = [
    ("Overview", overview_tab),
    ("Trust Dynamics", trust_tab),
    ("Ray ML Segments", ml_tab),
]

sidebar = []

if HAS_ROW_DATA:
    top_devtypes = row_quadrants["DevType"].value_counts().head(15).index.tolist()
    top_countries = row_quadrants["Country"].value_counts().head(15).index.tolist()

    devtype_filter = pn.widgets.MultiChoice(
        name="Developer role",
        options=sorted(top_devtypes),
        value=[],
        placeholder="all top roles",
    )
    country_filter = pn.widgets.MultiChoice(
        name="Country",
        options=sorted(top_countries),
        value=[],
        placeholder="all top countries",
    )
    exp_filter = pn.widgets.RangeSlider(
        name="Years of work experience",
        start=0,
        end=50,
        value=(0, 50),
        step=1,
    )
    include_unknown_exp = pn.widgets.Checkbox(name="Include unknown experience", value=True)

    bound_filtered_kpis = pn.bind(
        filtered_kpis, devtype_filter, country_filter, exp_filter, include_unknown_exp
    )
    bound_filtered_quadrants = pn.bind(
        filtered_quadrant_chart, devtype_filter, country_filter, exp_filter, include_unknown_exp
    )
    bound_filtered_roles = pn.bind(
        filtered_role_chart, devtype_filter, country_filter, exp_filter, include_unknown_exp
    )
    bound_filtered_countries = pn.bind(
        filtered_country_chart, devtype_filter, country_filter, exp_filter, include_unknown_exp
    )

    sidebar = [
        pn.pane.Markdown("## Filters"),
        devtype_filter,
        country_filter,
        exp_filter,
        include_unknown_exp,
    ]
    tabs_content.append(
        (
            "Filtered Profile",
            pn.Column(
                bound_filtered_kpis,
                bound_filtered_quadrants,
                pn.Row(bound_filtered_roles, bound_filtered_countries),
                sizing_mode="stretch_width",
            ),
        )
    )

tabs = pn.Tabs(*tabs_content, dynamic=True)

dashboard = pn.template.FastListTemplate(
    title="AI Trust Paradox Dashboard",
    site="CMP_SC-8540",
    sidebar=sidebar,
    main=[kpi_panel(), tabs],
    accent_base_color="#c92a2a",
    header_background="#1f2933",
    theme_toggle=False,
)

pn.config.raw_css.append(
    """
    :root {
      --card-border: #e5e7eb;
      --muted: #667085;
    }
    .metric-strip {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin: 0 0 10px 0;
    }
    .metric-strip-ml {
      grid-template-columns: repeat(5, minmax(130px, 1fr));
    }
    .metric-card {
      background: white;
      border: 1px solid var(--card-border);
      border-top: 4px solid #2b7cbf;
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      min-height: 74px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric-value {
      color: #111827;
      font-size: 25px;
      line-height: 1.05;
      font-weight: 800;
      margin-top: 6px;
    }
    .metric-sub {
      color: var(--muted);
      font-size: 11px;
      margin-top: 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .kpi-card {
      min-width: 178px;
      background: white;
      border: 1px solid var(--card-border);
      border-top: 4px solid #2b7cbf;
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .kpi-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .kpi-value {
      color: #111827;
      font-size: 27px;
      line-height: 1.1;
      font-weight: 750;
      margin-top: 8px;
    }
    .kpi-sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .insight-strip {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 18px 20px;
      margin-bottom: 10px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .insight-strip h2 {
      margin: 4px 0 8px 0;
      font-size: 22px;
      line-height: 1.2;
      color: #111827;
    }
    .insight-strip p {
      margin: 0;
      color: #344054;
      max-width: 900px;
    }
    .eyebrow {
      color: #c92a2a;
      font-weight: 750;
      font-size: 12px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .mode-badge {
      display: inline-block;
      white-space: nowrap;
      border-radius: 999px;
      background: #fff4e6;
      color: #b35c00;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 750;
    }
    .scope-note {
      color: #667085;
      font-size: 12px;
      line-height: 1.35;
      max-width: 260px;
      margin-top: 8px;
    }
    .data-note,
    .model-note,
    .chart-caption {
      background: #f8fafc;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: #344054;
      padding: 12px 14px;
      font-size: 13px;
      line-height: 1.45;
    }
    .chart-card {
      background: white;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .chart-card img {
      width: 100%;
      height: auto;
      display: block;
    }
    .chart-caption {
      margin-top: 8px;
      background: #ffffff;
    }
    .cluster-card {
      background: white;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .cluster-title {
      color: #111827;
      font-size: 15px;
      font-weight: 750;
      margin-bottom: 6px;
    }
    .cluster-card p {
      color: #344054;
      margin: 0 0 10px 0;
      line-height: 1.4;
    }
    .cluster-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .cluster-stats span {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 999px;
      color: #334155;
      font-size: 12px;
      padding: 4px 8px;
    }
    .about-panel {
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: #344054;
      padding: 14px 16px;
      font-size: 13px;
      line-height: 1.42;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .about-panel h3 {
      color: #111827;
      margin: 0 0 10px 0;
      font-size: 16px;
    }
    .about-panel ul {
      margin: 8px 0 10px 18px;
      padding: 0;
    }
    .about-panel li {
      margin-bottom: 6px;
    }
    .small-note {
      color: #667085;
      font-size: 12px;
      margin-bottom: 0;
    }
    @media (max-width: 900px) {
      .metric-strip,
      .metric-strip-ml {
        grid-template-columns: repeat(2, minmax(130px, 1fr));
      }
      .insight-strip {
        display: block;
      }
      .scope-note {
        max-width: none;
      }
    }
    """
)

dashboard.servable()

if __name__ == "__main__":
    print("Ready. Run with: panel serve dashboard/app.py --show --autoreload --port 5006")
