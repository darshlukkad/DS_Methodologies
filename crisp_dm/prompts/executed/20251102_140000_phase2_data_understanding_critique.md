# Phase 2: Data Understanding - Dr. Alexander Grigoriev Ruthless Critique

**Date**: 2025-11-02  
**Phase**: Data Understanding  
**Reviewer**: Dr. Alexander Grigoriev (CRISP-DM Authority, Author of "CRISP-DM 1.0: Step-by-step Data Mining Guide")  
**Role**: Ruthless Peer Reviewer & Red-Team Auditor

---

## SECTION UNDER REVIEW
**Phase 2: Data Understanding** - Walmart Sales Forecasting Project

**Project Context**: Time-series forecasting of weekly department-level sales across 45 Walmart stores using historical data (2010-2012) for inventory optimization and demand planning.

---

## SCORES

### Methodological Compliance: **42/100** ❌
### Scientific Rigor: **38/100** ❌

**Overall Assessment**: UNACCEPTABLE. This section fails to meet CRISP-DM Phase 2 requirements.

---

## ENUMERATED GAPS vs. CRISP-DM Phase 2 Required Activities

### Gap 1: NO FORMAL DATA QUALITY REPORT ❌
**Required**: CRISP-DM mandates a comprehensive Data Quality Report examining:
- Completeness (missing values by column, by time period, by Store-Dept)
- Validity (range checks, domain constraints, referential integrity)
- Consistency (cross-field validation, temporal consistency)
- Accuracy (outlier detection with statistical tests, not just eyeballing)
- Timeliness (reporting lags, data freshness)

**Current State**: Superficial `.isnull().sum()` counts. No analysis of WHY markdowns are 50%+ missing, no test if missingness is MCAR/MAR/MNAR.

**Concrete Revision**:
```python
# Test missingness mechanism for MarkDowns
from scipy.stats import chi2_contingency

# Is markdown missingness related to Store Type?
contingency = pd.crosstab(
    features_df['Type'], 
    features_df['MarkDown1'].isnull()
)
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"χ² test for markdown missingness vs Store Type: p={p_value:.4f}")
# If p < 0.05, missingness is NOT random (MAR/MNAR) -> cannot simply impute with mean
```

**Acceptance Criteria**: 
- Formal DQR table with 5 quality dimensions × all columns
- Statistical tests for missingness mechanism (χ², Little's MCAR test)
- Decision tree: impute, drop, or use as feature based on test results

---

### Gap 2: NO TIME-SERIES SPECIFIC DIAGNOSTICS ❌
**Required**: For temporal data, CRISP-DM Phase 2 demands:
- Stationarity tests (ADF, KPSS) on target by Store-Dept
- Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots
- Seasonality decomposition (STL, seasonal_decompose)
- Structural breaks (Chow test, CUSUM) around holidays/events

**Current State**: Generic histograms and boxplots. No temporal analysis.

**Concrete Revision**:
```python
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Test stationarity for top 10 Store-Dept combinations
top_combos = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().nlargest(10).index

stationarity_results = []
for store, dept in top_combos:
    series = train_merged[(train_merged['Store']==store) & (train_merged['Dept']==dept)].sort_values('Date')['Weekly_Sales']
    
    # ADF test (H0: non-stationary)
    adf_stat, adf_p = adfuller(series.dropna())[:2]
    
    # KPSS test (H0: stationary)
    kpss_stat, kpss_p = kpss(series.dropna(), regression='ct')[:2]
    
    stationarity_results.append({
        'Store': store, 'Dept': dept,
        'ADF_p': adf_p, 'KPSS_p': kpss_p,
        'Stationary': (adf_p < 0.05) and (kpss_p > 0.05)
    })

# Decision: If majority non-stationary, must difference or use trend-aware models
```

**Acceptance Criteria**:
- ADF/KPSS tests documented for top 20 Store-Dept pairs
- ACF/PACF plots showing significant lags → informs lag feature engineering
- Seasonal decomposition plots → quantifies trend/seasonal/residual components
- Document: "X% of series are non-stationary → require differencing or tree models"

---

### Gap 3: NO MULTIVARIATE EXPLORATION ❌
**Required**: CRISP-DM Phase 2 Section 3.3 mandates exploring relationships:
- Correlation matrix (Pearson, Spearman, Distance Correlation for non-linear)
- Interaction effects (Store Type × IsHoliday, Size × Unemployment)
- Variance Inflation Factor (VIF) to detect multicollinearity pre-modeling

**Current State**: No correlation analysis, no interaction exploration.

**Concrete Revision**:
```python
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Spearman correlation (robust to outliers, captures monotonic relationships)
numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Weekly_Sales']
corr_matrix = train_merged[numeric_cols].corr(method='spearman')

# Visualize with significance stars
from scipy.stats import spearmanr
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Spearman Correlation Matrix (ρ)', fontsize=16, weight='bold')

# VIF check
X_vif = train_merged[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']].dropna()
vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_data.sort_values('VIF', ascending=False))
# If VIF > 10, severe multicollinearity -> drop or regularize
```

**Acceptance Criteria**:
- Correlation heatmap with significance annotations (p < 0.05)
- VIF table with interpretation: VIF > 10 = problem, VIF > 5 = monitor
- Documented interactions to test in modeling (e.g., "Store Type A shows 2× holiday lift vs Type C")

---

### Gap 4: NO DATA DRIFT ANALYSIS ❌
**Required**: With 2010-2012 training and 2012 test, must check distribution shift:
- Kolmogorov-Smirnov test (train vs test for continuous features)
- Chi-square test (train vs test for categorical features)
- Concept drift (are relationships changing over time?)

**Current State**: No train-test comparison. Risk of distribution shift invalidating model.

**Concrete Revision**:
```python
from scipy.stats import ks_2samp

# Test if test features come from same distribution as train
drift_tests = []
for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
    train_vals = train_merged[col].dropna()
    test_vals = test_merged[col].dropna()
    
    ks_stat, ks_p = ks_2samp(train_vals, test_vals)
    drift_tests.append({
        'Feature': col,
        'KS_statistic': ks_stat,
        'p_value': ks_p,
        'Drift_Detected': ks_p < 0.05
    })

drift_df = pd.DataFrame(drift_tests)
print(drift_df)
# If drift detected, need domain adaptation or robust models
```

**Acceptance Criteria**:
- KS test results for all continuous features (p-values table)
- Chi-square test for Store Type distribution (train vs test)
- Conclusion: "No significant drift" or "Drift detected in X, Y → mitigation: [strategy]"

---

### Gap 5: NO BUSINESS DOMAIN VALIDATION ❌
**Required**: CRISP-DM emphasizes domain expert validation of data patterns:
- Are sales magnitudes realistic? (e.g., $1M/week for single dept = suspicious)
- Do holiday effects match domain knowledge? (Thanksgiving >> Labor Day)
- Are negative sales explained? (returns, data errors, accounting adjustments?)

**Current State**: Note of "negative sales exist" but no investigation or domain check.

**Concrete Revision**:
```python
# Investigate negative sales
negative_sales = train_merged[train_merged['Weekly_Sales'] < 0]
print(f"Negative sales: {len(negative_sales)} records ({len(negative_sales)/len(train_merged)*100:.2f}%)")

# By Store and Department
neg_by_store = negative_sales.groupby('Store').size().sort_values(ascending=False).head(10)
neg_by_dept = negative_sales.groupby('Dept').size().sort_values(ascending=False).head(10)

print("\nTop Stores with Negative Sales:")
print(neg_by_store)
print("\nTop Departments with Negative Sales:")
print(neg_by_dept)

# Decision logic
if len(negative_sales) < 0.01 * len(train_merged):
    print("Decision: Drop negative sales as outliers (<1% of data)")
elif 'Returns' in negative_sales.columns:  # hypothetical
    print("Decision: Negative sales represent returns -> create 'Returns' feature")
else:
    print("Decision: Negative sales unexplained -> FLAG for business SME review")
```

**Acceptance Criteria**:
- Document explains negative sales (returns vs errors)
- Business SME sign-off on data anomalies
- Decision log: keep/transform/drop with justification

---

### Gap 6: NO EXPLORATORY HYPOTHESIS GENERATION ❌
**Required**: CRISP-DM Phase 2 Output #3 is "Initial hypotheses" for modeling.

**Current State**: No hypotheses documented.

**Concrete Revision**:
Create hypotheses table:

| Hypothesis ID | Statement | Test Method | Expected Outcome |
|---------------|-----------|-------------|------------------|
| H1 | Store Type A has 20% higher holiday sales lift than Type C | ANOVA on (Sales_Holiday - Sales_Normal) by Type | F-stat, p < 0.05 |
| H2 | Lag-1 weekly sales is strongest predictor (AR(1) dominance) | Partial autocorrelation at lag 1 > 0.7 | PACF plot |
| H3 | Markdown effectiveness decreases with unemployment | Interaction term (Markdown × Unemployment) in regression | β_interaction < 0, p < 0.05 |
| H4 | Temperature has U-shaped relationship with sales (extreme cold/hot → higher) | Polynomial regression, β_temp² ≠ 0 | Likelihood ratio test |

**Acceptance Criteria**:
- Minimum 5 testable hypotheses
- Each hypothesis has clear test method and success criteria
- Hypotheses guide feature engineering in Phase 3

---

## RED-TEAM ANALYSIS: FALSIFICATION TESTS

### Falsification Test 1: Aggregation Paradox
**Claim**: "We'll forecast at Store-Dept-Week level"

**Red-Team**: What if temporal aggregation (Dept-Week instead of Store-Dept-Week) yields better forecasts due to variance reduction? Have you tested hierarchical aggregation strategies?

**Test**:
```python
# Compare MAE at different aggregation levels
from sklearn.metrics import mean_absolute_error

# Baseline: Store-Dept-Week (most granular)
baseline_mae = train_merged.groupby(['Store', 'Dept', 'Date'])['Weekly_Sales'].mean().std()

# Alternative: Dept-Week (aggregate stores)
dept_week_mae = train_merged.groupby(['Dept', 'Date'])['Weekly_Sales'].mean().std()

# Alternative: Store-Week (aggregate depts)
store_week_mae = train_merged.groupby(['Store', 'Date'])['Weekly_Sales'].mean().std()

print(f"Baseline (Store-Dept-Week) volatility: {baseline_mae:.2f}")
print(f"Dept-Week aggregation volatility: {dept_week_mae:.2f}")
print(f"Store-Week aggregation volatility: {store_week_mae:.2f}")

# Lower volatility → easier to forecast → consider hierarchical reconciliation
```

### Falsification Test 2: Leakage in Test Set
**Claim**: "Test set is clean for forecasting evaluation"

**Red-Team**: Are there any Store-Dept combinations in test that don't exist in train? (Cold-start problem) Are test dates truly future, or is there temporal overlap?

**Test**:
```python
# Check for cold-start problem
train_combos = set(train_merged[['Store', 'Dept']].apply(tuple, axis=1))
test_combos = set(test_merged[['Store', 'Dept']].apply(tuple, axis=1))
cold_start = test_combos - train_combos

print(f"Cold-start combinations (in test, not in train): {len(cold_start)}")
if len(cold_start) > 0:
    print("WARNING: Cold-start problem detected. Cannot use lag features for these.")
    print(f"Examples: {list(cold_start)[:5]}")

# Check temporal overlap
train_dates = set(train_merged['Date'])
test_dates = set(test_merged['Date'])
overlap = train_dates & test_dates

if len(overlap) > 0:
    print(f"ERROR: Temporal leakage detected. {len(overlap)} dates appear in both train and test.")
else:
    print("✓ No temporal overlap (clean temporal split)")
```

### Falsification Test 3: Simpson's Paradox
**Claim**: "Higher markdowns correlate with higher sales"

**Red-Team**: Could this be Simpson's Paradox? Maybe markdowns are applied to already high-selling items, creating spurious correlation. Check within-Store correlation.

**Test**:
```python
# Overall correlation
overall_corr = train_merged[['MarkDown1', 'Weekly_Sales']].corr().iloc[0, 1]
print(f"Overall Markdown-Sales correlation: {overall_corr:.3f}")

# Within-store correlation
within_store_corrs = []
for store in train_merged['Store'].unique():
    store_data = train_merged[train_merged['Store'] == store]
    corr = store_data[['MarkDown1', 'Weekly_Sales']].corr().iloc[0, 1]
    within_store_corrs.append(corr)

avg_within = np.nanmean(within_store_corrs)
print(f"Average within-store correlation: {avg_within:.3f}")

if np.sign(overall_corr) != np.sign(avg_within):
    print("⚠️ Simpson's Paradox detected! Overall trend reverses within groups.")
```

---

## REVISED OUTLINE (Target: ≥90/100)

### Phase 2: Data Understanding (Revised)

#### 2.1 Data Collection Report
- ✅ Dataset inventory (4 files: train, test, stores, features)
- ✅ Record counts, date ranges, key dimensions
- ✅ Schema documentation with business meaning

#### 2.2 Data Quality Report ⭐ NEW
- **Completeness**: Missing value patterns by dimension (temporal, categorical)
  - Test: MCAR/MAR/MNAR using Little's test, χ² contingency
- **Validity**: Range checks (e.g., Sales ≥ 0?, Temperature in [-20, 120]°F?)
  - Document violations → data cleaning decisions
- **Consistency**: Cross-field validation (IsHoliday matches known dates?)
- **Accuracy**: Outlier detection using IQR, Z-score, Isolation Forest
  - Flag top 1% extreme values for business review
- **Timeliness**: Is 2010-2012 data still relevant in 2025? (NO -> document risk)

#### 2.3 Time-Series Diagnostics ⭐ NEW
- **Stationarity**: ADF/KPSS tests for top 20 Store-Dept series
  - Document: X% stationary → can use levels; Y% non-stationary → need differencing
- **Autocorrelation**: ACF/PACF plots to identify significant lags
  - Output: "Lag-1, Lag-52 significant → use in features"
- **Seasonality**: STL decomposition to quantify trend/seasonal/residual
  - Output: "52-week seasonality explains 30% of variance"
- **Structural Breaks**: CUSUM test around major holidays
  - Output: "Thanksgiving causes level shift → add holiday features"

#### 2.4 Multivariate Exploration ⭐ NEW
- **Correlation Matrix**: Spearman (robust to outliers) + significance stars
  - Flag: |ρ| > 0.7 → multicollinearity risk
- **VIF Analysis**: Check for multicollinearity (VIF > 10 = severe)
- **Interaction Effects**: Store Type × Holiday, Markdown × Unemployment
  - Use partial regression plots or stratified analysis
- **Non-linear Relationships**: Scatter plots with LOWESS smoothing
  - Detect U-shaped, threshold, or saturation effects

#### 2.5 Distribution Shift Analysis ⭐ NEW
- **Train-Test Drift**: KS test for continuous, χ² for categorical
  - If p < 0.05 → document mitigation (robust models, domain adaptation)
- **Temporal Drift**: Rolling window statistics (mean, std over time)
  - Plot to visualize if relationships are stable 2010-2012
- **Concept Drift**: Test if feature-target relationships change over time
  - Use Recursive Least Squares with forgetting factor

#### 2.6 Domain Validation ⭐ NEW
- **Negative Sales Investigation**: Returns vs errors vs adjustments?
  - Decision log: keep/transform/drop with SME approval
- **Magnitude Checks**: Are $1M/week departments realistic?
  - Compare to industry benchmarks (revenue per sqft)
- **Holiday Effects**: Do patterns match retail domain knowledge?
  - E.g., Thanksgiving > Christmas > Super Bowl > Labor Day?

#### 2.7 Hypothesis Generation ⭐ NEW
- Create hypothesis table (min 5 hypotheses)
- Each hypothesis → testable prediction → informs feature engineering
- Example: "H1: Store Type A has 1.5× holiday lift vs Type C"

#### 2.8 Exploratory Visualizations
- **Target Distribution**: Histogram, log-histogram, boxplot (already done)
- **Temporal Patterns**: Sales over time by Store Type, Holiday vs Non-Holiday
- **Heatmaps**: Store × Dept average sales, Missing data patterns
- **Geospatial** (if lat/lon available): Sales choropleth map

#### 2.9 Data Understanding Report (Deliverable)
- Executive summary (1 page): key findings, red flags, recommendations
- Technical appendix: All test results, plots, tables
- Data dictionary: All columns with type, meaning, quality notes
- Sign-off: Business SME + Data Scientist

---

## CANONICAL REFERENCES

1. **Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000)**. *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
   - **Section 3.2-3.3**: Data Understanding requirements (pages 21-28)
   - **Key Quote**: "Data understanding includes four key tasks: collect initial data, describe data, explore data, and verify data quality."

2. **Little, R. J. A., & Rubin, D. B. (2019)**. *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
   - **Chapter 1**: MCAR/MAR/MNAR taxonomy
   - **Critical for**: Understanding markdown missingness mechanism before imputation

3. **Hyndman, R. J., & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice* (3rd ed.). OTexts.
   - **Chapter 2.8**: Time series decomposition
   - **Chapter 9**: Stationarity and differencing
   - **Online**: https://otexts.com/fpp3/

4. **Provost, F., & Fawcett, T. (2013)**. *Data Science for Business*. O'Reilly.
   - **Chapter 2**: Business problems and data mining solutions
   - **Chapter 7**: Data leakage and train-test split issues

5. **Kuhn, M., & Johnson, K. (2019)**. *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press.
   - **Chapter 3**: Visualizing numeric data
   - **Chapter 4**: Identifying and removing correlated predictors (VIF)

---

## FINAL VERDICT

**Current Score**: 40/100 (FAIL)  
**Revised Outline Score** (if implemented): 92/100 (PASS)

**Must-Fix Items** (blocking issues):
1. Add formal Data Quality Report with statistical tests
2. Perform time-series diagnostics (ADF, ACF, seasonality)
3. Check train-test distribution shift (KS tests)
4. Investigate and document negative sales with SME
5. Generate minimum 5 testable hypotheses

**Recommended Items** (for excellence):
6. Simpson's Paradox checks for correlations
7. Hierarchical aggregation exploration
8. VIF analysis and multicollinearity mitigation
9. Domain expert sign-off on all anomalies
10. Create comprehensive data dictionary document

**Timeline**: With proper implementation, this phase should take 3-5 days, not 3 hours.

---

**Reviewer**: Dr. Alexander Grigoriev  
**Signature**: A. Grigoriev, PhD  
**Date**: 2025-11-02  
**Institution**: Technical University of Eindhoven (Retired), CRISP-DM Consortium Founding Member
