# Critic-Driven Development Workflow

## Overview

This portfolio uses a **ruthless peer review** approach where world-renowned methodology experts critique each phase BEFORE proceeding. This ensures:

1. **Methodological compliance**: Adhering to official CRISP-DM/SEMMA/KDD standards
2. **Scientific rigor**: Statistical soundness, no p-hacking, proper validation
3. **Production readiness**: Real-world applicability, not academic exercises

---

## Expert Personas

### Dr. Alexander Grigoriev (CRISP-DM)
- **Authority**: Technical University of Eindhoven, CRISP-DM Consortium Founding Member
- **Expertise**: CRISP-DM methodology, industrial data mining, logistics optimization
- **Style**: Exacting, systematic, focuses on compliance with CRISP-DM 1.0 specification
- **Key Concerns**:
  - Are all required outputs documented?
  - Is the business problem clearly defined with decision framework?
  - Are baselines established before modeling?
  - Is data quality formally assessed (not just `.isnull().sum()`)?

### Dr. Eleanor Miner (SEMMA)
- **Authority**: SAS Institute (retired), Co-author of SEMMA methodology
- **Expertise**: Statistical modeling, SAS programming, pharmaceutical analytics
- **Style**: Statistical rigor, emphasizes exploratory analysis and model diagnostics
- **Key Concerns**:
  - Is sampling strategy justified?
  - Are distributional assumptions tested?
  - Is multicollinearity addressed (VIF)?
  - Are residuals examined for patterns?

### Dr. Usama Fayyad (KDD)
- **Authority**: UC Irvine, Co-founder of KDD Conference, Former Yahoo Chief Data Officer
- **Expertise**: Knowledge discovery, pattern mining, large-scale data systems
- **Expertise**: Feature selection, pattern evaluation, deployment at scale
- **Key Concerns**:
  - Is the knowledge discovery goal clear (descriptive vs predictive)?
  - Are patterns interpretable and actionable?
  - Is the model production-ready (latency, scalability)?
  - Are results validated with domain experts?

---

## Critique Structure

Each critique follows this format:

### 1. SCORES (0-100)
- **Methodological Compliance**: Adherence to official methodology phases
- **Scientific Rigor**: Statistical soundness, reproducibility, validity

### 2. ENUMERATED GAPS
For each gap:
- **Gap description**: What's missing vs required activities
- **Current state**: What was actually done (if anything)
- **Concrete revision**: Specific code/approach to fix (not generic advice)
- **Acceptance criteria**: How to verify the fix (testable)

### 3. RED-TEAM ANALYSIS
Falsification tests:
- Challenge core assumptions
- Test for common pitfalls (leakage, Simpson's Paradox, aggregation paradox)
- Provide executable test code

### 4. REVISED OUTLINE
- Complete outline that would score ≥90/100
- Mark NEW sections vs enhanced existing
- Prioritize must-fix vs nice-to-have

### 5. CANONICAL REFERENCES
- 3-5 authoritative sources
- Specific chapter/page numbers
- Key quotes justifying critique points

---

## Workflow Process

### Step 1: Implement Phase (Draft)
- Complete phase in notebook
- Use modular functions from `src/`
- Include visualizations and interpretations

### Step 2: Self-Review
- Compare against master prompt (e.g., `00_master_prompt.md`)
- Check if all required deliverables are present
- Run initial tests (e.g., leakage check, data quality)

### Step 3: Request Expert Critique
Save prompt to `prompts/executed/YYYYMMDD_HHMMSS_phaseX_critique_request.md`:

```markdown
# Critique Request: Phase X - [Phase Name]

**Date**: 2025-11-02
**Reviewer**: Dr. [Expert Name]

## Section Under Review
[Brief description of what was implemented]

## Project Context
[1-2 sentences: dataset, goal, methodology]

## Specific Questions
1. [Question 1]
2. [Question 2]
3. [Question 3]

## Artifacts
- Notebook cells: [cell numbers]
- Reports: [file names]
- Code modules: [file names]

[Attach relevant notebook sections or copy key code blocks]
```

### Step 4: Receive Ruthless Critique
Expert provides:
- Scores (typically 30-60/100 on first draft)
- 5-10 enumerated gaps with concrete fixes
- 2-3 red-team falsification tests
- Revised outline scoring ≥90/100

Save to `prompts/executed/YYYYMMDD_HHMMSS_phaseX_critique_response.md`

### Step 5: Implement Fixes
For each gap:
1. Understand the required fix
2. Implement concrete revision (use provided code as template)
3. Verify acceptance criteria met
4. Document in notebook

### Step 6: Document Actions Taken
In notebook, add "Critique Response" section:

```markdown
## 📝 Critique Response & Actions Taken

**Date**: 2025-11-02
**Phase**: [Phase Name]
**Reviewer**: Dr. [Expert Name]
**Initial Score**: 42/100
**Post-Fix Score**: 91/100

### Summary of Critique
[2-3 sentences summarizing main issues]

### Actions Implemented

#### Fix 1: [Gap Name]
- **Issue**: [Description]
- **Action**: [What you did]
- **Code**: [Link to notebook cell or module]
- **Verification**: [How you tested it]

#### Fix 2: [Gap Name]
...

### Outstanding Items
[Any items deferred with justification]
```

### Step 7: Iterate (if needed)
- If score < 85/100, request follow-up review
- If score ≥ 85/100, proceed to next phase

---

## Example: Business Understanding Critique

### Initial Implementation (Score: 58/100)
```python
# Business understanding
business_understanding = {
    'project': 'Walmart Weekly Sales Forecasting',
    'objective': 'Predict department-level sales for inventory optimization',
    'stakeholders': ['Operations', 'Supply Chain', 'Marketing', 'Finance'],
    # ... basic KPIs ...
}
```

### Critique: Gap 1 - NO FORMAL DECISION ANALYSIS ❌
**Fix**: Add decision framework
```python
# Enhanced business understanding
business_understanding = {
    # ... existing fields ...
    'decision_framework': {
        'primary_decision_maker': {
            'role': 'Regional Inventory Manager (Operations)',
            'name': 'Sarah Chen',
            'decision': 'Weekly inventory replenishment quantities',
            'action_space': 'Order 0 to 50,000 units per Store-Dept',
            'timing': 'Every Monday 9 AM for following week',
            'decision_rule': 'IF forecast > 0.7 × inventory: order = forecast - inventory + safety_stock'
        },
        'model_integration': 'Forecast feeds SAP ERP → auto-generates POs'
    }
}
```

### Post-Fix (Score: 93/100) ✅
- Decision framework added with named decision-maker
- Action space and timing specified
- Decision rule explicitly uses model output

---

## Quality Gates

Before proceeding to next phase:

| Phase | Min Score | Key Deliverables | Sign-Off |
|-------|-----------|------------------|----------|
| Business Understanding | 85/100 | Project charter, ROI, FMEA, baselines | Business sponsor |
| Data Understanding | 85/100 | DQR, EDA report, hypotheses table | Data + Business SME |
| Data Preparation | 90/100 | Processed data, leakage tests, pipeline | Data scientist + reviewer |
| Modeling | 90/100 | Model comparison, CV results, SHAP | Data scientist + stakeholder UAT |
| Evaluation | 90/100 | Holdout test, segment analysis, recommendations | All stakeholders |
| Deployment | 85/100 | API, monitoring, handoff docs | Ops/DevOps + business |

---

## Benefits of This Approach

### 1. Prevents Common Pitfalls
- Data leakage caught in Data Prep phase
- P-hacking avoided via pre-registered hypotheses
- Overfitting detected through rigorous CV

### 2. Ensures Production Readiness
- Business value quantified (ROI) before building
- Edge cases identified early (cold-start, drift)
- Monitoring strategy defined upfront

### 3. Builds Trust with Stakeholders
- Systematic approach demonstrates professionalism
- External "expert review" adds credibility
- Clear documentation enables handoff

### 4. Educational Value
- Critiques teach advanced techniques (VIF, FMEA, KS tests)
- References point to authoritative sources
- Revised outlines serve as templates for future projects

---

## Anti-Patterns to Avoid

### ❌ Generic Praise
**Bad**: "Great job on EDA! Visualizations look nice."
**Good**: "Distribution analysis is superficial. Missing: (1) KS test for train-test drift (p=0.03 indicates shift), (2) ADF test for stationarity (72% of series non-stationary → need differencing), (3) VIF analysis (CPI and Unemployment have VIF=12 → multicollinearity)."

### ❌ Vague Recommendations
**Bad**: "Consider adding more features."
**Good**: "Add interaction terms: Store_Type × IsHoliday. Hypothesis: Type A has 1.5× holiday lift vs Type C. Test with ANOVA: F(2,421037)=?, p<0.05? If significant, include in model."

### ❌ Accepting Mediocrity
**Bad**: "Score 55/100 is passing. Proceed to next phase."
**Good**: "Score 55/100 is FAILURE. Must-fix items: (1) Add ROI calculation, (2) Measure baselines, (3) Create FMEA table. Estimated time: 1 day. Do not proceed until score ≥85/100."

---

## Tools & Resources

### Statistical Tests (Python)
```python
from scipy.stats import chi2_contingency, ks_2samp, spearmanr
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### Quality Checks
- **Data Leakage**: `check_data_leakage(train, val, test)` in `src/feature_engineering.py`
- **Baseline Performance**: `naive_baseline_*` functions in `src/modeling.py`
- **Model Comparison**: MLflow tracking with consistent metrics

### Documentation Templates
- `prompts/00_master_prompt.md` - Phase requirements
- `prompts/critic_persona.md` - Reviewer background
- `prompts/executed/*.md` - Timestamped critiques

---

## Conclusion

This ruthless peer review process is **the difference between a portfolio project and production-ready data science**. 

It forces:
- Methodological rigor
- Statistical soundness  
- Business alignment
- Production readiness

The critiques may seem harsh, but they teach advanced techniques that separate junior from senior data scientists.

**Remember**: "In theory, there is no difference between theory and practice. In practice, there is." - Yogi Berra

---

**Last Updated**: 2025-11-02
**Status**: Active workflow for all three methodologies (CRISP-DM, SEMMA, KDD)
