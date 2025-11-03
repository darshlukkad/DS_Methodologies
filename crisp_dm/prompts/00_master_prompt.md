# CRISP-DM Master Prompt: Walmart Sales Forecasting

## Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)

### Objective
Implement a complete CRISP-DM workflow for Walmart Sales Forecasting, demonstrating industry best practices in time-series forecasting with a production-ready approach.

## Project Context

**Dataset**: Walmart Recruiting - Store Sales Forecasting (Kaggle)
**Business Problem**: Predict weekly department-level sales across 45 Walmart stores
**Time Period**: Feb 2010 - Oct 2012 (historical training), Nov 2012 (forecasting horizon)

## CRISP-DM Phases

### 1. Business Understanding
**Goals**:
- Define clear business objectives and success criteria
- Identify key stakeholders and their needs
- Establish KPIs and evaluation metrics
- Analyze costs and benefits
- Document risks and assumptions

**Deliverables**:
- Business understanding document
- KPI definitions (MAE, sMAPE, WAPE)
- Baseline models (naive last-week, last-year-same-week)
- Cost-benefit analysis
- Risk assessment (holiday/promo leakage, overfitting)

### 2. Data Understanding
**Goals**:
- Collect and describe data sources
- Explore data quality and distributions
- Identify data quality issues
- Generate initial hypotheses

**Deliverables**:
- Data dictionary with all features
- Exploratory data analysis (EDA) visualizations
- Data quality report
- Summary statistics
- Correlation analysis

### 3. Data Preparation
**Goals**:
- Clean and transform data
- Handle missing values and outliers
- Create features for modeling
- Implement time-aware splits (no data leakage)

**Deliverables**:
- Time-aware train/validation/test splits
- Feature engineering: date parts, holiday flags, (Store,Dept) lags/rolling (1,2,4,52 weeks)
- Promotion flags and interaction features
- Preprocessing pipeline (sklearn Pipeline)
- Leakage checks and tests

### 4. Modeling
**Goals**:
- Select and train multiple algorithms
- Tune hyperparameters systematically
- Validate with time-series cross-validation
- Track experiments with MLflow

**Models**:
- Baseline: Naive (last week, last year same week)
- Linear: Ridge, Lasso
- Tree-based: Random Forest, XGBoost, LightGBM

**Deliverables**:
- Trained models with hyperparameters
- MLflow experiment logs
- Cross-validation results
- Model comparison table
- SHAP explanations (global + local)
- Per-segment error analysis (Store, Department, Holiday weeks)

### 5. Evaluation
**Goals**:
- Assess model performance on holdout test set
- Compare against baselines and business requirements
- Analyze errors by segment and time period
- Test stability and robustness

**Deliverables**:
- Final evaluation report
- Performance vs baselines
- Segment analysis (high/low volume stores, holiday weeks)
- Sensitivity analysis
- Business impact translation
- Recommendations

### 6. Deployment
**Goals**:
- Prepare model for production
- Create API endpoint
- Implement monitoring strategy

**Deliverables**:
- Serialized model (joblib)
- FastAPI `/predict` endpoint with full pipeline
- Input validation with Pydantic
- Evidently drift detection report
- Monitoring plan document
- Docker deployment instructions

## Success Criteria

### Technical Metrics
- **MAE** < 2,000 units (industry benchmark)
- **sMAPE** < 15% (symmetric mean absolute percentage error)
- **WAPE** < 12% (weighted absolute percentage error)
- Beat naive baselines by at least 20%

### Business Metrics
- Improved inventory allocation
- Reduced stockouts and overstock
- Better promotional planning
- ROI: Cost savings > model development/maintenance costs

## Key Considerations

### Data Leakage Prevention
- Strict time-aware splits (no future information in past)
- No target leakage in features
- Proper handling of holidays and promotions
- Cross-validation respects temporal order (TimeSeriesSplit)

### Feature Engineering
- **Temporal**: day of week, month, quarter, week of year
- **Lags**: 1, 2, 4, 52 weeks (handle seasonality)
- **Rolling**: mean/std over 4, 8, 52 weeks
- **Events**: holiday flags, markdown events
- **Entity**: store/department fixed effects

### Model Interpretability
- SHAP force plots for individual predictions
- SHAP summary plots for global understanding
- Feature importance rankings
- Partial dependence plots for key features

### Production Readiness
- Reproducible preprocessing pipeline
- Version control for data and models
- API with proper validation and error handling
- Monitoring for data drift and model performance
- Documentation for maintenance

## Critic Loop

After each phase, engage **Dr. Cassie Kozyrkov** (Chief Decision Scientist, Google) to critique:
- Clarity of business problem definition
- Rigor of statistical methods
- Quality of decision-making framework
- Practical applicability
- Risk identification

Document all critiques and actions taken in `prompts/executed/`.

## Quality Standards

- **Code**: Modular, tested, type-hinted, documented
- **Notebooks**: Runnable top-to-bottom, well-structured, reproducible
- **Documentation**: Clear, comprehensive, stakeholder-appropriate
- **Testing**: Unit tests for critical functions, integration tests for pipeline
- **Version Control**: Track data versions, model versions, code versions

## Deliverable Format

Single Jupyter notebook (`CRISP_DM.ipynb`) with:
- Clear section headers for each phase
- Code in functions (from `src/` modules)
- Markdown explanations before each major step
- Visualizations with insights
- Critic cells after each phase
- Links to external artifacts (reports, models, tests)

---

**Created**: 2025-11-02  
**Status**: Active  
**Methodology**: CRISP-DM v1.0
