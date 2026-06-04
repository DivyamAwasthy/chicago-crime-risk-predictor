# Chicago Crime Risk Predictor

An end-to-end machine-learning system that predicts crime **risk level** 
(Low / Medium / High) for any location and time in Chicago — from raw open 
data through to a deployed, interactive web app.

**🔗 Live demo:** https://chicago-crime-risk-predictor-4qqsuld2x893evqgpqclpx.streamlit.app

---

## Overview

This is the supervised-classification half of a two-part urban safety platform.
The companion project, **[SafeRoute]([(https://github.com/Yashwardhan2/safe-route))**, 
tackles the same Chicago crime data from the routing angle (unsupervised hotspot 
clustering + graph-based safe-path routing). Together they form a 
**Chicago Urban Safety Platform** with three components:

- **Unsupervised** — HDBSCAN crime-hotspot detection (SafeRoute)
- **Supervised** — XGBoost spatiotemporal risk classification (this project)
- **Graph routing** — modified Dijkstra crime-aware pathfinding (SafeRoute)

This repo covers the supervised component: predicting how risky a given 
(location, time) cell is, with model explainability and a deployed demo.

---

## Data

- **Source:** [Chicago Open Data Portal](https://data.cityofchicago.org/resource/ijzp-q8t2.csv), crimes from 2022 onward
- **Volume:** ~300k records
- **Cleaning:** dropped null coordinates (~1,443 rows, <0.5%); filtered to 
  Chicago bounds (lat 41.6–42.1, lon −87.95 to −87.5)
- **Temporal features:** hour, day_of_week, month, is_weekend
- **Spatial grid:** lat/lon rounded to 2 decimals (~1 km cells) → 78,494 unique bins
- **Target:** crime count per (cell, time) bucketed via `qcut` into 3 balanced 
  classes — Low (47.1%), Medium (24.5%), High (28.4%)

---

## Model & Results

Features: `[lat_bin, lon_bin, hour, day_of_week, is_weekend]` · 
80/20 stratified split (train 62,795 / test 15,699)

| Model | Accuracy | F1 Macro | F1 High | F1 Medium | F1 Low |
|-------|:--------:|:--------:|:-------:|:---------:|:------:|
| Logistic Regression (baseline) | 0.495 | 0.342 | 0.389 | 0.000 | 0.637 |
| XGBoost | 0.646 | 0.546 | 0.695 | 0.185 | 0.760 |
| **XGBoost + SMOTE (deployed)** | **0.631** | **0.581** | **0.688** | **0.317** | **0.738** |

**The deployed model is XGBoost + SMOTE.** SMOTE was chosen as the production 
model despite a ~1.5% accuracy drop because it nearly **doubles Medium-class F1 
(0.185 → 0.317)** — the hardest class, since "medium-risk" cells sit in the 
overlap region between clearly-safe and clearly-dangerous areas. For a 
risk-screening tool, balanced performance across all three classes matters more 
than raw accuracy driven by the majority class.

### Live behaviour (demo points)

The model discriminates strongly by both location and time:

| Location | Time | Prediction | Confidence |
|----------|------|:----------:|:----------:|
| South Chicago (41.70, −87.62) | Sat, 00:00 | High | 71% |
| North Chicago (41.98, −87.68) | Tue, 05:00 | Low | 97% |
| Central (41.85, −87.65) | Sat, 14:00 | Medium | 43% |
| West side (41.88, −87.72) | Fri, 20:00 | High | 89% |

---

## Explainability (SHAP)

TreeExplainer on [FILL: 2000] samples:

- **`lat_bin` and `lon_bin` dominate**, ~2× the impact of `hour` — location is 
  the primary risk driver
- `day_of_week` and `is_weekend` have near-zero impact
- For the High class, southern-Chicago coordinates push predictions toward 
  high risk; northern coordinates push toward low risk

---

## Key Design Decisions

- **~1 km grid (2-decimal rounding):** balances spatial resolution against having 
  enough crimes per cell for a stable target
- **`qcut` over fixed thresholds:** data-driven, produces balanced classes
- **Dropped `arrest` / `domestic` features:** these are only known *after* a crime 
  occurs — including them would be data leakage
- **Logistic Regression baseline:** establishes a performance floor; its failure 
  on the Medium class (F1 = 0.000) shows linear boundaries can't carve the middle class
- **XGBoost over Random Forest:** boosting's sequential error-correction outperformed 
  bagging on this overlapping-class problem
- **`max_depth=6`:** controls overfitting on the tree ensemble
- **SMOTE for deployment:** accepts a small accuracy cost to recover minority-class recall

---

## Architecture

Chicago Open Data → cleaning + feature engineering ([`01_eda.ipynb`](01_eda.ipynb)) → 
spatiotemporal aggregation → model training + SHAP ([`02_modeling.ipynb`](02_modeling.ipynb)) → 
Streamlit app ([`app.py`](app.py))

The deployed app **trains the model on startup** from the aggregated CSV (cached via 
`@st.cache_resource`) rather than loading a pickle — this avoids 
Python/library-version mismatches between the training and deployment environments.

---

## Repository

| File | Description |
|------|-------------|
| `01_eda.ipynb` | Data loading, cleaning, temporal features, EDA |
| `02_modeling.ipynb` | Baseline, XGBoost, SMOTE, SHAP analysis |
| `crime_risk_aggregated.csv` | Aggregated per-cell training data (78,494 rows) |
| `app.py` | Streamlit app (trains on startup, serves predictions) |
| `requirements.txt` | Pinned dependencies |

## Tech Stack

Python 3.12 · pandas · scikit-learn 1.6.1 · XGBoost 3.2.0 · 
imbalanced-learn 0.14.1 · SHAP · Streamlit

---

## Limitations & Future Work

- Medium-class F1 (0.317) remains the weakest — the class is intrinsically hard 
  due to feature-space overlap
- Coordinate features are treated numerically; finer spatial encoding 
  (e.g. neighbourhood embeddings) could help
- Temporal signal is weak relative to spatial; richer time features 
  (holidays, seasonality) are a natural extension
