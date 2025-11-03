Title: CRISP-DM in Practice — 5-6 Minute Video Script

Intro (0:00–0:20)
Hi, I’m Darsh, In this short video I’ll show how we used the CRISP‑DM methodology to build a production-ready weekly sales forecasting system for Walmart. I’ll cover the six CRISP‑DM phases, key technical decisions we made, and the business impact — all in five minutes.

Hook (0:20–0:35)
We reduced forecast error and translated that into an estimated $2.1 million in annual savings by focusing not just on models, but on process, checks, and production readiness.

Phase 1 — Business Understanding (0:35–0:55)
Start by asking: what decision will this model enable? For us, it was inventory allocation and promotional planning. We defined concrete KPIs: sMAPE, MAE, and a business ROI target. Documenting acceptance criteria avoided chasing vanity metrics later.

Phase 2 — Data Understanding (0:55–1:20)
Next, we profiled the data. We ran a 5‑dimension Data Quality Report for completeness, validity, consistency, accuracy, and timeliness. We discovered missing markdown fields and negative sales. After SME consultation we treated negatives as returns and kept them as features.

Phase 3 — Data Preparation & Feature Engineering (1:20–2:00)
This is where models get their power. We engineered leakage-safe lag features (t-1, t-52), backward-looking rolling stats, holiday indicators, and markdown aggregates. Key rule: features must only use past data. We also ensured the test set had matching features — missing ones were filled with safe defaults and logged.

Phase 4 — Modeling (2:00–2:35)
We start with baselines — last-week and last-year — then simple linear models (Ridge) and move to tree ensembles (XGBoost, LightGBM). Time-aware splits and a small CV plan ensured we didn’t overfit. LightGBM delivered the best results.

Phase 5 — Evaluation (2:35–3:15)
Evaluation includes segment analysis (store type, dept, holidays), residual diagnostics, and business translation — converting error reduction into dollars. We also validated our hypotheses statistically (e.g., lag-52 seasonality). The best model had WMAE ~2,512 and met our business targets.

Phase 6 — Deployment & Monitoring (3:15–3:50)
We wrapped model + preprocessing into a FastAPI app, containerized it, and deployed with monitoring (Evidently + custom KS/PSI checks). We set alert thresholds for drift and performance degradation and scheduled quarterly retraining.

Lessons & Tips (3:50–4:30)
1. Document the business decision first. Metrics must map to dollars.  
2. Prevent leakage at every step — especially with time series.  
3. Start with simple baselines and move to complexity only when needed.  
4. Build observability (drift, KPI monitoring) from day one.

Call to Action (4:30–4:50)
If you want the full notebook, code, and visual reports, check the GitHub repo linked below. I also wrote a Medium article that walks through the whole process step-by-step — link in the README.

Closing (4:50–5:00)
Thanks for watching — if you found this useful, like and subscribe for more practical ML engineering walkthroughs. Questions? Drop them in the comments.
