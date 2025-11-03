Title: SEMMA Talk — Sample, Explore, Modify, Model, Assess (5–6 Minute Script)

Intro (0:00–0:18)
Hi — I’m Darsh, In the next five minutes I’ll walk through how we applied the SEMMA process to the weekly sales forecasting problem. SEMMA emphasizes a clear path from sampling data to assessing models — perfect for reproducible applied data science.

Hook (0:18–0:35)
This project shows that careful sampling and exploration, combined with disciplined feature modification, produce reliable models. We used SEMMA to cut experimental time while increasing reproducibility.

Sample (0:35–1:05)
We sampled at the Store-Dept level, ensuring representation across high- and low-volume pairs and across promotional seasons. Stratified temporal sampling was used: we reserved the latest 15% of weeks as validation to keep the time order. For reproducibility we store the sampling seed and exact indices used.

Explore (1:05–1:45)
EDA focused on label behavior, seasonality, and missingness. Visual diagnostics included time-series decomposition, autocorrelation plots, and grouped histograms. Important discoveries: shipment/markdown fields showed bursty patterns and returns created negative-sales tails. We documented these in a discovery notebook and flagged features for careful imputation.

Modify (1:45–2:25)
Feature engineering followed deterministic, unit-tested transforms: lag features, rolling stats, holiday encodings, and markdown aggregates. We also normalized numerical features per-store quantiles and encoded categories with ordinal encoders trained only on the train sample. All transformations are pipeline objects so the exact transform is recorded with the model.

Model (2:25–3:05)
We trained and compared Ridge, XGBoost, and LightGBM using the sampled train split. Hyperparameter searches were small, grid-based, and reproducible (random seeds logged). The model pipeline includes preprocessing, feature selection, and the estimator. We measured both pointwise error (MAE, RMSE) and business-aligned metrics (sMAPE, WAPE).

Assess (3:05–3:45)
Assessment included holdout evaluation, segment-wise error analysis, stability checks across time slices, and ablation tests. We also checked model explainability with SHAP summaries for top features. LightGBM offered the best combination of accuracy and inference speed.

Reproducibility & Artifacts (3:45–4:15)
All sampling code, EDA notebooks, and pipeline transforms are in the repository. The README includes explicit commands to recreate the sampling and experiment runs. We also provide seed-controlled experiment logs and a Dockerfile for the scoring runtime.

Deployment & Monitoring (4:15–4:45)
After validation, the pipeline is containerized and exposed as a FastAPI endpoint with logging. Monitoring includes drift detection (PSI), performance checks, and weekly retraining triggers when error increases beyond thresholds.

Closing & CTA (4:45–5:00)
If you want to reproduce any sampling or modeling step, run `crisp_dm/CRISP_DM.ipynb` and consult the `crisp_dm/src` preprocessing and modeling modules. Links and a reproducibility checklist are in the repository. Thanks — leave requests for code examples or deeper results in the comments.
