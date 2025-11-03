Title: KDD Talk — Knowledge Discovery in Practice (5–6 Minute Script)

Intro (0:00–0:20)
Hi — I’m Darsh . In the next five minutes I’ll present how we applied Knowledge Discovery in Databases (KDD) principles to build a robust weekly sales forecasting pipeline. I’ll cover the research question, our data and feature strategy, model choices, evaluation, and the reproducibility and deployment steps we took to make the work production-ready.

Hook (0:20–0:40)
This was more than a modelling exercise — by treating discovery as a repeatable, documented pipeline we reduced forecasting error meaningfully and turned model improvements into tangible operational value.

Problem & Dataset (0:40–1:05)
Problem: predict weekly sales at store-department granularity to inform inventory and promotion planning. Dataset: historical sales, promotions/markdowns, store metadata, and calendar/holiday features. The data posed typical KDD challenges: missing values, returns (negative sales), irregular seasonality, and categorical heterogeneity across stores.

Exploratory Discovery (1:05–1:40)
We started by asking: what patterns are real and which are artifacts? Visualizations and grouped statistics revealed strong weekly and annual seasonality, prominent holiday effects, and frequent sparse-skewed behavior for smaller store-department pairs. Importantly, exploratory checks discovered negative weekly sales that represented returns — this changed our label handling and created new features (return counts and magnitudes).

Feature Strategy & Leakage Control (1:40–2:20)
KDD emphasizes defensible features — we built lag features (t-1, t-52), rolling statistics (4, 8, 52-week windows), holiday indicators, and markdown aggregates. The key rule: features are strictly backward-looking — no peeking into the future. For reproducibility, each feature transformation is a deterministic function versioned in our preprocessing module and tested on historical slices.

Modeling & Evaluation (2:20–3:05)
We compared baselines (last-week, last-year), linear baselines (Ridge), and ensembles (XGBoost, LightGBM). Because KDD is about discovery, we ran targeted ablations: which lag set matters most? Do markdown aggregates help? The evaluation used time-aware splits and a small rolling-window CV. Our metrics: sMAPE and WAPE for business-aligned error, plus MAE and RMSE for interpretability. LightGBM consistently beat baselines on held-out periods.

Key Findings (3:05–3:35)
Top takeaways: (1) simple lag and rolling features capture most signal; (2) holiday and markdown aggregates give consistent uplift; (3) careful handling of returns reduces bias in error estimates. Ablation studies showed diminishing returns after adding complex interaction features, suggesting the best trade-off is in robust preprocessing and ensembling.

Reproducibility & Artifacts (3:35–4:05)
All code, preprocessing functions, and model recipes are in the repository. We include a runnable notebook, seed-controlled experiments, and a small Dockerized FastAPI app that scores new data. The paper-style deliverable includes experiment tables, hyperparameters, and the exact git commit used for each reported result.

Deployment & Monitoring (4:05–4:35)
To move from discovery to production, we containerized the preprocessing + model, deployed behind a small API, and added monitoring for data drift, performance degradation, and failure cases. Alerts are configured for sudden PSI/KL shifts and sMAPE breaches on recent windows.

Closing & Call to Action (4:35–5:00)
If you want to reproduce any result, run the `CRISP_DM.ipynb` from the repository’s `crisp_dm/` folder — links and a step-by-step README are there. For a deeper read, check the Medium article linked in the repo. Thanks — leave questions or requests for code samples in the comments.
