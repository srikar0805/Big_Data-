"""
Phase 2 - Step 05 - Visualisations + smoke tests.

Generates static PNGs from the derived Spark SQL and Ray artefacts in output/.
This keeps the final visualisation step lightweight and reproducible from the
files committed with the project, while scripts 01-04 remain the source of the
underlying data preparation, analytics, and ML outputs.

Reads  : output/spark_sql_results/*
         output/ray_ml_results/*
Writes : output/visualizations/*.png
"""
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
SPARK_SQL = OUTPUT_DIR / "spark_sql_results"
ML = OUTPUT_DIR / "ray_ml_results"
VIZ = OUTPUT_DIR / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d9d9d9",
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

BLUE = "#2b7cbf"
GREEN = "#2f9e44"
ORANGE = "#f08c00"
RED = "#c92a2a"
GRAY = "#6c757d"
PURPLE = "#7950f2"

QUADRANT_COLORS = {
    "High Usage - High Trust": GREEN,
    "High Usage - Low Trust": RED,
    "Low Usage - High Trust": BLUE,
    "Low Usage - Low Trust": GRAY,
}

ROLE_LABELS = {
    "Developer, AI apps or physical AI": "AI app developer",
    "AI/ML engineer": "AI/ML engineer",
    "Developer, embedded applications or devices": "Embedded developer",
    "Developer, game or graphics": "Game/graphics developer",
}

CLUSTER_LABELS = {
    0: "0 Power users: high usage + high trust",
    1: "1 Frustrated adopters: high concern",
    2: "2 Regular low-friction users",
    3: "3 Low adoption / low trust",
}

USAGE_LABELS = {
    "Yes, I use AI tools daily": "Daily",
    "Yes, I use AI tools weekly": "Weekly",
    "Yes, I use AI tools monthly or infrequently": "Monthly or infrequent",
    "No, but I plan to soon": "Plan to soon",
    "No, and I don't plan to": "Do not plan to",
    "Unknown": "Unknown",
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


def spark_csv(name):
    files = sorted((SPARK_SQL / name).glob("part-*.csv"))
    if not files:
        raise FileNotFoundError(f"Missing Spark SQL CSV for {name}")
    return pd.read_csv(files[0])


def wrap(value, width=28):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def clean_role(value):
    return ROLE_LABELS.get(value, value.replace("Developer, ", "").replace(" or ", " / "))


def savefig(name):
    plt.tight_layout()
    plt.savefig(VIZ / name, dpi=150, bbox_inches="tight")
    plt.close()


def fmt_count(value, _pos=None):
    return f"{int(value):,}"


def label_hbars(ax, values, pad=0.01, fmt="{:,.0f}"):
    xmax = max(values) if len(values) else 0
    offset = xmax * pad if xmax else 1
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_width() + offset,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )


def label_points(ax, xs, ys, labels, dy=0.12):
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y + dy, label, ha="center", va="bottom", fontsize=9, color="#333333")


def add_note(fig, text):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8.5, color="#666666")


def ordered_experience(df):
    return (
        df.assign(_order=df["ExperienceGroup"].map({v: i for i, v in enumerate(EXPERIENCE_ORDER)}))
          .sort_values("_order")
          .drop(columns="_order")
    )


start = time.time()

dataset_summary = spark_csv("dataset_summary")
ai_usage = spark_csv("ai_usage_distribution")
trust_by_usage = spark_csv("trust_by_usage")
trust_by_role = spark_csv("trust_by_role")
experience = ordered_experience(spark_csv("experience_analysis"))
quadrants = spark_csv("quadrant_summary")
paradox_profile = spark_csv("paradox_profile")

rf_metrics = json.loads((ML / "rf_metrics.json").read_text())
feature_importance = pd.read_csv(ML / "paradox_feature_importance.csv")
ray_clusters = pd.read_csv(ML / "ray_cluster_results.csv")
cluster_summary = pd.read_csv(ML / "cluster_summary.csv")
kmeans_scan = pd.read_csv(ML / "kmeans_scan.csv")

total_trust_rows = int(dataset_summary.loc[0, "RespondentsWithTrust"])

# -----------------------------------------------------------------------------
# Viz 1 - AI usage frequency distribution
# -----------------------------------------------------------------------------
usage = ai_usage.copy()
usage["Label"] = usage["AISelect"].map(USAGE_LABELS).fillna(usage["AISelect"])
usage = usage.sort_values("DeveloperCount")

fig, ax = plt.subplots(figsize=(9, 5.2))
colors = [GRAY if label == "Unknown" else BLUE for label in usage["Label"]]
ax.barh(usage["Label"], usage["DeveloperCount"], color=colors)
ax.set_title("AI Tool Usage Frequency")
ax.set_xlabel("Developers")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(FuncFormatter(fmt_count))
label_hbars(ax, usage["DeveloperCount"].to_numpy())
savefig("ai_usage_distribution.png")

# -----------------------------------------------------------------------------
# Viz 2 - Trust and frustration by usage band
# -----------------------------------------------------------------------------
band_order = ["Low (0-25)", "Medium-Low (25-50)", "Medium-High (50-75)", "High (75-100)"]
usage_band = (
    trust_by_usage.assign(_order=trust_by_usage["UsageBand"].map({v: i for i, v in enumerate(band_order)}))
                  .sort_values("_order")
                  .drop(columns="_order")
)
x = np.arange(len(usage_band))

fig, ax = plt.subplots(figsize=(9, 5.3))
ax.plot(x, usage_band["AvgTrustScore"], marker="o", lw=2.5, color=GREEN, label="Trust")
ax.plot(x, usage_band["AvgFrustrationScore"], marker="o", lw=2.5, color=ORANGE, label="Frustration")
label_points(ax, x, usage_band["AvgTrustScore"], [f"{v:.1f}" for v in usage_band["AvgTrustScore"]], dy=1.0)
label_points(ax, x, usage_band["AvgFrustrationScore"], [f"{v:.1f}" for v in usage_band["AvgFrustrationScore"]], dy=1.0)
ax.set_xticks(x)
ax.set_xticklabels([wrap(v, 16) for v in usage_band["UsageBand"]])
ax.set_ylabel("Average score (0-100)")
ax.set_ylim(0, 82)
ax.set_title("Trust Rises With AI Usage, but Frustration Also Rises")
ax.legend(loc="upper left")
for i, n in enumerate(usage_band["DeveloperCount"]):
    ax.text(i, 2.0, f"n={n:,}", ha="center", va="bottom", fontsize=8.5, color="#666666")
savefig("trust_by_usage.png")

# -----------------------------------------------------------------------------
# Viz 3 - Composite UsageScore vs TrustScore
# -----------------------------------------------------------------------------
sample = ray_clusters[["UsageScore", "TrustScore", "FrustrationScore"]].dropna()
if len(sample) > 12000:
    sample = sample.sample(12000, random_state=0)
rng = np.random.default_rng(0)
sx = sample["UsageScore"] + rng.uniform(-0.45, 0.45, len(sample))
sy = sample["TrustScore"] + rng.uniform(-0.45, 0.45, len(sample))

fig, ax = plt.subplots(figsize=(8, 6.8))
sc = ax.scatter(sx, sy, c=sample["FrustrationScore"], cmap="plasma", s=9, alpha=0.38, linewidths=0)
fig.colorbar(sc, ax=ax, label="Frustration score")
median_usage = rf_metrics["median_usage"]
median_trust = rf_metrics["median_trust"]
ax.axvline(median_usage, ls="--", color="#222222", lw=1, alpha=0.65)
ax.axhline(median_trust, ls="--", color="#222222", lw=1, alpha=0.65)
ax.text(3, 96, "Low usage\nhigh trust", ha="left", va="top", fontsize=9, color=BLUE)
ax.text(97, 96, "High usage\nhigh trust", ha="right", va="top", fontsize=9, color=GREEN)
ax.text(97, 4, "High usage\nlow trust", ha="right", va="bottom", fontsize=9, color=RED)
ax.text(3, 4, "Low usage\nlow trust", ha="left", va="bottom", fontsize=9, color=GRAY)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Usage score")
ax.set_ylabel("Trust score")
ax.set_title("Usage and Trust Form Four Clear Quadrants")
add_note(fig, f"Dashed lines are medians: Usage {median_usage:.1f}, Trust {median_trust:.1f}.")
savefig("usage_vs_trust_scatter.png")

# -----------------------------------------------------------------------------
# Viz 4 - Quadrant distribution
# -----------------------------------------------------------------------------
quad_order = [
    "Low Usage - Low Trust",
    "Low Usage - High Trust",
    "High Usage - Low Trust",
    "High Usage - High Trust",
]
quad = (
    quadrants.assign(_order=quadrants["TrustUsageGroup"].map({v: i for i, v in enumerate(quad_order)}))
             .sort_values("_order")
             .drop(columns="_order")
)

fig, ax = plt.subplots(figsize=(9, 5.4))
colors = [QUADRANT_COLORS[g] for g in quad["TrustUsageGroup"]]
ax.barh([wrap(g, 24) for g in quad["TrustUsageGroup"]], quad["DeveloperCount"], color=colors)
ax.set_xlabel("Developers")
ax.set_ylabel("")
ax.set_title("Trust-Usage Quadrant Sizes")
ax.xaxis.set_major_formatter(FuncFormatter(fmt_count))
shares = quad["DeveloperCount"] / total_trust_rows * 100
label_hbars(ax, quad["DeveloperCount"].to_numpy(), fmt="{:,.0f}")
for patch, share in zip(ax.patches, shares):
    ax.text(
        patch.get_width() * 0.02,
        patch.get_y() + patch.get_height() / 2,
        f"{share:.1f}%",
        va="center",
        ha="left",
        fontsize=9,
        color="white",
        weight="bold",
    )
add_note(fig, "Quadrants are split at the median UsageScore and median TrustScore among respondents with full trust data.")
savefig("quadrant_distribution.png")

# -----------------------------------------------------------------------------
# Viz 5 - Frustration by trust-usage group
# -----------------------------------------------------------------------------
frust = quad.sort_values("AvgFrustrationScore")
fig, ax = plt.subplots(figsize=(9, 5.2))
colors = [QUADRANT_COLORS[g] for g in frust["TrustUsageGroup"]]
ax.barh([wrap(g, 24) for g in frust["TrustUsageGroup"]], frust["AvgFrustrationScore"], color=colors)
ax.set_xlabel("Average frustration score")
ax.set_ylabel("")
ax.set_xlim(24, 35.5)
ax.set_title("The Paradox Group Has the Highest Frustration")
label_hbars(ax, frust["AvgFrustrationScore"].to_numpy(), fmt="{:.1f}")
add_note(fig, "Axis is intentionally narrowed to show the difference between quadrant averages.")
savefig("frustration_by_group.png")

# -----------------------------------------------------------------------------
# Viz 6 - Trust by experience group
# -----------------------------------------------------------------------------
exp = experience.copy()
exp["ShortGroup"] = exp["ExperienceGroup"].map(EXPERIENCE_SHORT)
x = np.arange(len(exp))
known = exp[exp["ExperienceGroup"] != "Unknown"]
known_range = known["AvgTrustScore"].max() - known["AvgTrustScore"].min()
all_range = exp["AvgTrustScore"].max() - exp["AvgTrustScore"].min()

fig, ax = plt.subplots(figsize=(9, 5.4))
ax.plot(x, exp["AvgTrustScore"], marker="o", lw=2.5, color=GREEN)
ax.fill_between(x, exp["AvgTrustScore"], exp["AvgTrustScore"].min(), color=GREEN, alpha=0.08)
label_points(ax, x, exp["AvgTrustScore"], [f"{v:.1f}" for v in exp["AvgTrustScore"]], dy=0.08)
ax.set_xticks(x)
ax.set_xticklabels(exp["ShortGroup"], rotation=0)
ax.set_ylabel("Average trust score")
ax.set_title("Years of Experience Barely Move AI Trust")
ax.set_ylim(exp["AvgTrustScore"].min() - 0.8, exp["AvgTrustScore"].max() + 0.8)
for i, n in enumerate(exp["DeveloperCount"]):
    ax.text(i, ax.get_ylim()[0] + 0.08, f"n={n:,}", ha="center", va="bottom", fontsize=8.2, color="#666666")
add_note(
    fig,
    f"Known-experience groups vary by only {known_range:.1f} points; including Unknown, the range is {all_range:.1f} points.",
)
savefig("experience_vs_trust.png")

# -----------------------------------------------------------------------------
# Viz 7 - Heatmap of binned UsageScore x TrustScore counts
# -----------------------------------------------------------------------------
df_h = ray_clusters[["UsageScore", "TrustScore"]].dropna().copy()
df_h["u_bin"] = (df_h["UsageScore"] // 10).clip(upper=9).astype(int) * 10
df_h["t_bin"] = (df_h["TrustScore"] // 10).clip(upper=9).astype(int) * 10
heat = (
    df_h.groupby(["t_bin", "u_bin"]).size()
        .unstack(fill_value=0)
        .reindex(index=range(0, 100, 10), columns=range(0, 100, 10), fill_value=0)
)

fig, ax = plt.subplots(figsize=(7.6, 6.4))
im = ax.imshow(heat.values, aspect="auto", cmap="viridis", origin="lower", extent=[0, 100, 0, 100])
fig.colorbar(im, ax=ax, label="Developers")
ax.axvline(median_usage, ls="--", color="white", lw=1.2, alpha=0.8)
ax.axhline(median_trust, ls="--", color="white", lw=1.2, alpha=0.8)
ax.set_xlabel("Usage score, binned by 10")
ax.set_ylabel("Trust score, binned by 10")
ax.set_title("Where Developers Cluster by Usage and Trust")
savefig("usage_trust_heatmap.png")

# -----------------------------------------------------------------------------
# Viz 8 - Random-Forest feature importance
# -----------------------------------------------------------------------------
friendly_features = {
    "WorkExpNum": "Work experience",
    "WorkflowIntegration": "Workflow integration",
    "AIModelCount": "AI model count",
    "DevEnvToolCount": "Dev environment tools",
    "AgentDepth": "Agent usage depth",
    "UsageFreq": "AI usage frequency",
    "ProblemCount": "AI problem count",
    "ThreatLevel": "AI threat perception",
}
fi = feature_importance.copy()
fi["FeatureLabel"] = fi["Feature"].map(friendly_features).fillna(fi["Feature"])
fi = fi.sort_values("Importance")

fig, ax = plt.subplots(figsize=(8.8, 5.2))
ax.barh(fi["FeatureLabel"], fi["Importance"] * 100, color=PURPLE)
ax.set_xlabel("Mean importance (%)")
ax.set_ylabel("")
ax.set_title("What Predicts the High-Usage, Low-Trust Group?")
label_hbars(ax, (fi["Importance"] * 100).to_numpy(), fmt="{:.1f}%")
add_note(fig, f"Random-forest ensemble accuracy: {rf_metrics['accuracy']:.1%}. Trust fields are excluded from predictors.")
savefig("rf_feature_importance.png")

# -----------------------------------------------------------------------------
# Viz 9 - Ray clusters in PCA space
# -----------------------------------------------------------------------------
cluster_features = [
    "UsageScore", "TrustScore", "FrustrationScore",
    "UsageFreq", "AgentDepth", "AIModelCount", "WorkflowIntegration",
    "ProblemCount", "ThreatLevel", "DevEnvToolCount", "WorkExpNum",
]
cluster_sample = ray_clusters.dropna(subset=cluster_features + ["Cluster"]).copy()
if len(cluster_sample) > 9000:
    cluster_sample = cluster_sample.sample(9000, random_state=0)
Xc = StandardScaler().fit_transform(cluster_sample[cluster_features].values)
pcs = PCA(n_components=2, random_state=0).fit_transform(Xc)
cluster_sample["PC1"] = pcs[:, 0]
cluster_sample["PC2"] = pcs[:, 1]

fig, ax = plt.subplots(figsize=(10.5, 6.2))
for cluster_id in sorted(cluster_sample["Cluster"].unique()):
    rows = cluster_sample[cluster_sample["Cluster"] == cluster_id]
    ax.scatter(rows["PC1"], rows["PC2"], s=8, alpha=0.45, label=CLUSTER_LABELS.get(int(cluster_id), f"Cluster {int(cluster_id)}"))
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Developer Segments From Ray K-Means")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), markerscale=2)
savefig("cluster_pca.png")

# -----------------------------------------------------------------------------
# Viz 10 - Role trust extremes
# -----------------------------------------------------------------------------
roles = trust_by_role.copy()
roles["RoleLabel"] = roles["DevType"].map(clean_role)
top_roles = roles.nlargest(8, "AvgTrustScore")
bottom_roles = roles.nsmallest(5, "AvgTrustScore")
role_extremes = (
    pd.concat([bottom_roles, top_roles])
      .drop_duplicates(subset=["DevType"])
      .sort_values("AvgTrustScore")
)
role_colors = [RED if v < roles["AvgTrustScore"].median() else GREEN for v in role_extremes["AvgTrustScore"]]

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh([wrap(v, 30) for v in role_extremes["RoleLabel"]], role_extremes["AvgTrustScore"], color=role_colors)
ax.set_xlabel("Average trust score")
ax.set_ylabel("")
ax.set_xlim(38, 68.5)
ax.set_title("Roles With the Highest and Lowest AI Trust")
label_hbars(ax, role_extremes["AvgTrustScore"].to_numpy(), fmt="{:.1f}")
add_note(fig, "Includes roles with at least 100 developers. AI app developers lead; game/graphics and embedded developers are lowest.")
savefig("role_trust_extremes.png")

# -----------------------------------------------------------------------------
# Viz 11 - Country-wise paradox profile
# -----------------------------------------------------------------------------
remote_labels = {
    "Remote": "Remote",
    "In-person": "In-person",
    "Unknown": "Unknown",
    "Hybrid (some remote, leans heavy to in-person)": "Hybrid",
    "Hybrid (some in-person, leans heavy to flexibility)": "Hybrid",
    "Your choice (very flexible, you can come in when you want or just as needed)": "Flexible",
}
country = paradox_profile.copy()
country["RemoteSimple"] = country["RemoteWork"].map(remote_labels).fillna("Other")
country_totals = (
    country.groupby("Country", as_index=False)["DeveloperCount"].sum()
           .sort_values("DeveloperCount", ascending=False)
           .head(8)
)
country_top = country[country["Country"].isin(country_totals["Country"])]
pivot = (
    country_top.pivot_table(
        index="Country", columns="RemoteSimple", values="DeveloperCount", aggfunc="sum", fill_value=0
    )
    .reindex(country_totals.sort_values("DeveloperCount")["Country"])
)
remote_order = [c for c in ["Remote", "Hybrid", "Flexible", "In-person", "Unknown", "Other"] if c in pivot.columns]
remote_colors = {
    "Remote": BLUE,
    "Hybrid": ORANGE,
    "Flexible": PURPLE,
    "In-person": GRAY,
    "Unknown": "#adb5bd",
    "Other": "#495057",
}

fig, ax = plt.subplots(figsize=(9, 5.6))
left = np.zeros(len(pivot))
for remote in remote_order:
    values = pivot[remote].to_numpy()
    ax.barh(pivot.index, values, left=left, label=remote, color=remote_colors[remote])
    left += values
ax.set_xlabel("Paradox-profile developers")
ax.set_ylabel("")
ax.set_title("Country Breakdown of the High-Usage, Low-Trust Profile")
ax.xaxis.set_major_formatter(FuncFormatter(fmt_count))
for i, total in enumerate(left):
    ax.text(total + max(left) * 0.01, i, f"{int(total):,}", va="center", ha="left", fontsize=9)
ax.legend(loc="lower right")
add_note(fig, "Uses the Spark paradox-profile output: role x country x remote slices with at least 20 developers.")
savefig("country_paradox_profile.png")


# =============================================================================
# Testing
# =============================================================================
def run_tests():
    tests = []

    total = int(dataset_summary.loc[0, "TotalResponses"])
    tests.append(("T1 dataset row count", 40000 < total < 60000, f"got {total:,}"))

    required_outputs = {
        "dataset_summary": dataset_summary,
        "trust_by_usage": trust_by_usage,
        "trust_by_role": trust_by_role,
        "experience_analysis": experience,
        "quadrant_summary": quadrants,
        "paradox_profile": paradox_profile,
    }
    required_columns = {
        "dataset_summary": {"TotalResponses", "RespondentsWithTrust", "AvgTrustScore"},
        "trust_by_usage": {"UsageBand", "DeveloperCount", "AvgTrustScore", "AvgFrustrationScore"},
        "trust_by_role": {"DevType", "DeveloperCount", "AvgTrustScore"},
        "experience_analysis": {"ExperienceGroup", "DeveloperCount", "AvgTrustScore"},
        "quadrant_summary": {"TrustUsageGroup", "DeveloperCount", "AvgTrustScore", "AvgFrustrationScore"},
        "paradox_profile": {"DevType", "Country", "RemoteWork", "DeveloperCount"},
    }
    missing = {
        name: sorted(cols - set(frame.columns))
        for name, cols in required_columns.items()
        for frame in [required_outputs[name]]
        if not cols.issubset(frame.columns)
    }
    tests.append(("T2 expected result columns", not missing, f"missing: {missing}"))

    score_columns = [
        dataset_summary["AvgUsageScore"],
        dataset_summary["AvgTrustScore"],
        dataset_summary["AvgFrustrationScore"],
        trust_by_usage["AvgTrustScore"],
        trust_by_usage["AvgFrustrationScore"],
        trust_by_role["AvgTrustScore"],
        experience["AvgTrustScore"],
        quadrants["AvgTrustScore"],
        quadrants["AvgFrustrationScore"],
        ray_clusters["UsageScore"],
        ray_clusters["TrustScore"],
        ray_clusters["FrustrationScore"],
    ]
    all_scores = pd.concat(score_columns, ignore_index=True).dropna()
    tests.append(("T3 score bounds 0-100", all_scores.between(0, 100).all(), f"min={all_scores.min():.1f}, max={all_scores.max():.1f}"))

    highest_role = roles.sort_values("AvgTrustScore", ascending=False).iloc[0]
    lowest_role = roles.sort_values("AvgTrustScore", ascending=True).iloc[0]
    role_ok = (
        highest_role["DevType"] == "Developer, AI apps or physical AI"
        and lowest_role["DevType"] == "Developer, game or graphics"
    )
    tests.append((
        "T4 role trust ordering",
        role_ok,
        f"highest={highest_role['AvgTrustScore']:.1f} {highest_role['DevType']}; "
        f"lowest={lowest_role['AvgTrustScore']:.1f} {lowest_role['DevType']}",
    ))

    tests.append(("T5 known-experience trust range < 1 point", known_range < 1.0, f"range={known_range:.2f}"))

    expected = [
        VIZ / "ai_usage_distribution.png",
        VIZ / "trust_by_usage.png",
        VIZ / "usage_vs_trust_scatter.png",
        VIZ / "quadrant_distribution.png",
        VIZ / "frustration_by_group.png",
        VIZ / "experience_vs_trust.png",
        VIZ / "usage_trust_heatmap.png",
        VIZ / "rf_feature_importance.png",
        VIZ / "cluster_pca.png",
        VIZ / "role_trust_extremes.png",
        VIZ / "country_paradox_profile.png",
    ]
    missing_files = [str(p) for p in expected if not p.exists() or p.stat().st_size == 0]
    tests.append(("T6 output artefacts", not missing_files, f"missing: {missing_files}"))

    width = max(len(t[0]) for t in tests)
    for name, ok, detail in tests:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}  {detail}")
    return all(ok for _, ok, _ in tests)


print()
all_passed = run_tests()
elapsed = time.time() - start
print(f"\nGenerated {len(list(VIZ.glob('*.png')))} visualisation PNGs in {elapsed:.1f}s")
print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
