# Poster Content Outline (18x24 in) — AI Trust Paradox, Phase 2

## Title
**AI Trust Paradox, Phase 2: An Empirical Big Data Study of Trust in AI Coding Tools Using Apache Spark and Ray**

## Authors
**Preya Patel, Sai Srikar**  
CMP_SC-8540, Big Data and Model Management (Spring 2026)

## Abstract
This project analyzes whether heavier AI-tool usage among developers leads to higher trust or increased skepticism. Using the 2025 Stack Overflow Developer Survey (49,191 responses; 172 columns), we built composite **Usage**, **Trust**, and **Frustration** scores on a 0–100 scale from multiple survey columns. We processed data with Apache Spark, executed Spark SQL analytics, and used Ray to orchestrate K-Means clustering and Random Forest classification for paradox profiling. Results show that trust generally rises with usage, but a meaningful subgroup of high-usage developers still reports low trust (**High Usage–Low Trust**, 4,312 developers), with the highest frustration among all quadrants. The pipeline also provides role-, experience-, and country-level slices, 11 generated visualizations, and a reproducible dashboard for exploration.

## Introduction
- **Research question:** Do developers who use AI tools more frequently trust them more, or does heavy usage expose limitations and create skepticism?
- **Motivation:** AI coding assistants are widely adopted, but trust and reliability perceptions vary by developer segment.
- **Objective:** Build a reproducible big-data pipeline that quantifies usage/trust/frustration, identifies paradox segments, and explains drivers using distributed analytics and ML.

## Dataset
- **Source:** 2025 Stack Overflow Developer Survey (`survey_results_public.csv`)
- **Scale:** 49,191 rows × 172 columns (135 MB CSV)
- **Coverage from derived outputs:** 178 countries, 33 developer-role categories
- **Labeling logic:** Trust-based analyses use respondents with complete trust fields (27,258 respondents)
- **License:** ODbL (Open Database License)

## Methodology
1. **Ingestion & cleaning (Spark):** CSV loaded and standardized to Parquet.
2. **Feature engineering (Spark SQL):**
   - `UsageScore` from frequency, agent depth, model breadth, workflow integration
   - `TrustScore` from accuracy trust, complex-task trust, sentiment
   - `FrustrationScore` from frustration-problem count and perceived threat
3. **Analytics (Spark SQL):** usage-band summaries, trust-by-role, experience trends, and trust×usage quadrants.
4. **ML (Ray + scikit-learn):**
   - K-Means segmentation (4 clusters)
   - Random Forest paradox classifier (accuracy: **83.6%**)
5. **Visualization & validation:** 11 static PNG figures + 6 smoke tests (all passing in this repo run).

## Implementation / Repo Structure
- `scripts/01_data_ingestion_cleaning.py` → raw CSV to cleaned Parquet
- `scripts/02_feature_engineering_scores.py` → composite and component scores
- `scripts/03_spark_sql_analytics.py` → SQL aggregations and quadrant outputs
- `scripts/04_ray_machine_learning.py` → K-Means + Random Forest outputs
- `scripts/05_visualizations_testing.py` → figure generation + smoke tests
- `output/spark_sql_results/` → dataset/analytics CSV outputs
- `output/ray_ml_results/` → cluster summaries, RF metrics, feature importances
- `output/visualizations/` → 11 generated poster-ready PNG charts
- `dashboard/app.py` → interactive Panel dashboard

## Results
### Key findings
- Average scores (all respondents with available fields): **Usage 25.23**, **Trust 52.98**, **Frustration 19.63**.
- Trust increases across usage bands (30.1 → 74.4), while frustration also rises (9.9 → 34.9).
- Quadrant counts (median split):
  - High Usage–High Trust: 9,480
  - Low Usage–Low Trust: 8,671
  - Low Usage–High Trust: 4,795
  - **High Usage–Low Trust (Paradox): 4,312** (highest frustration: **33.1**)
- Highest-trust role: **Developer, AI apps or physical AI (66.2)**; lowest-trust role: **Developer, game or graphics (40.6)**.
- Experience effect is weak (known-experience trust spread < 1 point).
- Top paradox predictors (RF importance): **WorkExpNum (22.0%)**, **WorkflowIntegration (19.9%)**, **AIModelCount (15.1%)**.

### Figure list and captions (use these in poster)
1. **`output/visualizations/ai_usage_distribution.png`**  
   *Distribution of developers across AI usage score bins.*
2. **`output/visualizations/trust_by_usage.png`**  
   *Average trust and frustration by usage band; both increase with heavier usage.*
3. **`output/visualizations/usage_vs_trust_scatter.png`**  
   *Usage vs. trust relationship at respondent level with visible variability.*
4. **`output/visualizations/quadrant_distribution.png`**  
   *Counts across trust×usage quadrants highlighting the paradox group.*
5. **`output/visualizations/frustration_by_group.png`**  
   *Frustration comparison by quadrant; paradox group has highest frustration.*
6. **`output/visualizations/experience_vs_trust.png`**  
   *Trust by experience group showing near-flat trend.*
7. **`output/visualizations/usage_trust_heatmap.png`**  
   *Joint density view of usage and trust score regions.*
8. **`output/visualizations/rf_feature_importance.png`**  
   *Random Forest feature importances for paradox classification.*
9. **`output/visualizations/cluster_pca.png`**  
   *PCA projection of Ray K-Means developer segments.*
10. **`output/visualizations/role_trust_extremes.png`**  
    *Highest- and lowest-trust developer roles.*
11. **`output/visualizations/country_paradox_profile.png`**  
    *Country/work-mode paradox profile slices.*

## Conclusion
The project confirms a measurable **AI Trust Paradox**: high usage does not uniformly imply high trust. While trust tends to increase with usage overall, a substantial high-usage/low-trust segment persists and reports the greatest frustration. Role-based differences are stronger than experience-based differences, and ML models can identify paradox membership with strong performance.

## Future Work
- Validate generalization on future developer-survey releases.
- Add temporal drift analysis as AI tooling evolves.
- Expand explainability (e.g., SHAP) for paradox classification.
- Extend dashboard with interactive cohort comparisons and scenario filters.
- Scale pipeline to multi-node Spark/Ray environments.

## References
1. Stack Overflow Developer Survey 2025: https://survey.stackoverflow.co/
2. Apache Spark documentation: https://spark.apache.org/docs/latest/
3. Ray documentation: https://docs.ray.io/en/latest/
4. scikit-learn documentation: https://scikit-learn.org/stable/
5. Panel documentation: https://panel.holoviz.org/

## Acknowledgements
- Course faculty and mentors for project guidance in CMP_SC-8540.
- Stack Overflow for publishing the open developer survey dataset.
- Open-source communities behind Spark, Ray, scikit-learn, Matplotlib, and Panel.

---

### If figures are missing in another clone/environment
Regenerate all analytics and plots with:

```bash
pip install -r requirements.txt
mkdir -p logs
for s in scripts/0*.py; do
  name=$(basename "$s" .py)
  python "$s" > "logs/${name}.log" 2>&1
done
```

Then pull figures from `output/visualizations/*.png` for poster placement.
