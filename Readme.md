# Telecom RF Network Quality Intelligence System

An end-to-end analytics platform that ingests RF/telecom network data, engineers domain-specific signal quality features, classifies network degradation root causes using ML, and visualizes results through an interactive Tableau dashboard.

**Live Dashboard:** [View on Tableau Public](https://public.tableau.com/app/profile/jatin.maggo/viz/RFNetworkQualityIntelligenceSystem/RFNetworkQualityIntelligenceDashboard)

---

## Problem Statement

Telecom operators lose significant revenue to poor network quality — dropped calls, weak signal, congestion — but often lack visibility into *why* a specific tower or region is underperforming. This project builds a pipeline that:
1. Models real Indian cell tower geography in a PostgreSQL star schema
2. Engineers RF quality features from raw signal/call data
3. Classifies the root cause of network degradation (interference, congestion, distance, hardware fault) using ML
4. Forecasts call-drop probability ahead of time using an LSTM
5. Surfaces insights through an interactive geographic dashboard

---

## Datasets

| Dataset | Source | Use |
|---|---|---|
| Mobile Network Coverage India | OpenCelliD (via Kaggle) | 2.4M+ real Indian cell tower locations (lat/long, radio type, operator) |
| Cellular Network Performance | Kaggle | Call-level signal quality records (signal strength, SNR, distance, attenuation) |
| RF Signal Data | Kaggle | RF environmental readings (frequency, modulation, interference type, bandwidth) |

**Note on scale:** The primary ML fact table (call-level records) contains 462 rows. This is a real constraint of the available call-quality dataset. Feature engineering, class-imbalance handling, and model choices were made with this in mind — see "Honest Limitations" below.

---

## Architecture

```
Raw CSVs → Python ETL (pandas) → PostgreSQL star schema → ML models → Tableau
```

**Star Schema (PostgreSQL):**
- `dim_tower` — 2.4M+ tower geolocations (lat/long, radio type, MCC/MNC)
- `dim_operator` — operator + telecom circle mapping
- `dim_date` — date dimension
- `dim_environment` — call environment (urban/home/open)
- `fact_rf_readings` — engineered RF features + degradation labels

---

## Feature Engineering

The source datasets do not contain standard telecom KPIs directly, so four features were engineered:

| Feature | Engineering Approach |
|---|---|
| **RSRP proxy** | Derived from Signal Strength (dBm) — same physical unit/concept |
| **SINR proxy** | Derived from SNR — same underlying concept |
| **PRB Utilization** | Simulated from call duration + a bandwidth-driven load factor sampled from the RF signal dataset's real distribution |
| **Handover Rate** | Calculated per-user as the frequency of tower switches across their call sequence |

**Degradation cause labels** (interference / congestion / distance / hardware_fault) are **rule-based**, derived from combinations of distance, SINR, PRB utilization, and attenuation thresholds — since no public dataset provides ground-truth root-cause labels for network degradation. This is a common approach in exploratory telecom analytics projects; it enables a legitimate multi-class classification problem while being transparent about how the target was constructed.

---

## Machine Learning Pipeline

**1. XGBoost — Multi-class degradation classifier**
- SMOTE applied to handle class imbalance (hardware_fault had only 5 samples)
- Result: 86% overall accuracy; 90%+ precision/recall on congestion and distance classes
- Top predictive features: SINR proxy, attenuation, call duration — consistent with RF physics

**2. K-Means — Tower behavior clustering**
- 5 clusters identified (silhouette score 0.225)
- Labeled clusters: "Congested tower," "High mobility zone," etc., based on cluster-mean characteristics
- SHAP explainability applied to the XGBoost model to validate feature importance

**3. LSTM — 24-hour-ahead call-drop probability forecast**
- Since raw call records are sparse and not naturally hourly, an hourly time-series was constructed using the dataset's observed statistical distributions with realistic daily traffic seasonality (peak congestion during typical high-usage hours)
- Test MAE: 0.11 (on 0–1 scaled probability)

---

## Dashboard (Tableau)

- **Geo-Heatmap** — all towers plotted by lat/long, color-coded by degradation cause, with operator/circle tooltips
- **SPC Control Chart** — SINR trend over time with 3-sigma control bands to flag statistically anomalous signal drift

---

## Honest Limitations

This project was built under real dataset constraints, and transparency matters more than inflated claims:

- The ML fact table is small (462 rows) — a limitation of the available real call-quality dataset, not a design choice
- Degradation-cause labels are rule-based/derived, not ground-truth operator data
- Tower-to-reading assignment is randomized (no natural join key existed across the three source datasets)
- LSTM training data uses an engineered hourly series based on real observed distributions, not literal hour-by-hour logs

These constraints are typical of working with public, non-proprietary telecom datasets, and are documented here rather than hidden.

---

## Tech Stack

`Python (pandas, scikit-learn, XGBoost, SHAP, imbalanced-learn, TensorFlow/Keras) · PostgreSQL · SQLAlchemy · Tableau`

---

## Project Structure

```
rf_telecom/
├── data/
│   ├── raw/              # original Kaggle/OpenCelliD CSVs
│   └── processed/        # cleaned + engineered outputs
├── src/
│   ├── etl_pipeline.py
│   ├── load_to_postgres.py
│   ├── train_xgboost.py
│   ├── kmeans_shap.py
│   └── train_lstm.py
├── sql/
│   ├── analytical_queries.sql
│   └── export_for_tableau.sql
└── dashboards/            # Tableau workbook
```
