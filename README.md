# Data Mining Methodologies Portfolio

A production-quality data science portfolio demonstrating three industry-standard methodologies through complete, end-to-end implementations.

## 🎯 Overview

This portfolio showcases three classic data mining methodologies applied to real-world datasets:

1. **CRISP-DM** - Cross-Industry Standard Process for Data Mining (Walmart Sales Forecasting)
2. **SEMMA** - Sample, Explore, Modify, Model, Assess (Student Performance Prediction)
3. **KDD** - Knowledge Discovery in Databases (NSL-KDD Intrusion Detection)

Each methodology is implemented as a **single, comprehensive Jupyter notebook** covering the full data science lifecycle from business understanding to deployment.

## 📊 Methodology Implementations

### CRISP-DM: Walmart Sales Forecasting
**Dataset**: Walmart Recruiting Store Sales Forecasting  
**Business Goal**: Predict weekly department sales across stores  
**Key Features**:
- Time-aware train/validation/test splits
- Advanced feature engineering (lags, rolling windows, holiday flags)
- Multiple models (Ridge, Random Forest, XGBoost, LightGBM)
- SHAP explanations and segment analysis
- FastAPI deployment with monitoring

**Notebook**: [`crisp_dm/CRISP_DM.ipynb`](crisp_dm/CRISP_DM.ipynb)

### SEMMA: Student Performance Prediction
**Dataset**: Students Performance in Exams  
**Business Goal**: Predict student exam performance based on demographics and study habits  
**Key Features**:
- SAS implementation + Python mirror
- Comprehensive statistical exploration
- Feature engineering and selection
- Multiple classification models with ROC/PR/Lift curves
- Model interpretation and recommendations

**Notebook**: [`semma/SEMMA.ipynb`](semma/SEMMA.ipynb)

### KDD: NSL-KDD Intrusion Detection
**Dataset**: NSL-KDD Network Intrusion Dataset  
**Business Goal**: Detect and classify network intrusions  
**Key Features**:
- Multi-class intrusion classification
- Cost-sensitive learning (false negatives more expensive)
- Per-class performance metrics
- SHAP interpretation for security insights
- Optional API deployment

**Notebook**: [`kdd/KDD.ipynb`](kdd/KDD.ipynb)

## Articles & Video Walkthroughs

For readers who prefer written walkthroughs or video presentations, the following Medium articles and companion videos are available:

- CRISP-DM (Walmart Sales Forecasting)
	- Medium article: https://medium.com/@darshlukkad/crisp-dm-in-practice-a-hands-on-guide-to-reliable-data-science-projects-8f11d2f1ed1d
	- Video walkthrough: https://drive.google.com/file/d/1VGsgHQeLYUCTQoZtpEoD7qM-eBNBnOSE/view?usp=drive_link

- KDD (Knowledge Discovery in Databases)
	- Medium article: https://medium.com/@darshlukkad/knowledge-discovery-in-practice-a-reproducible-kdd-walkthrough-for-weekly-sales-forecasting-076969dcb444
	- Video walkthrough: https://drive.google.com/file/d/1rJORNx5LbKpLEeqOzB8LMV8KsIV4Zvav/view?usp=drive_link

- SEMMA (Sample, Explore, Modify, Model, Assess)
	- Medium article: https://medium.com/@darshlukkad/applying-semma-for-reproducible-weekly-sales-forecasting-cd954f58f664
	- Video walkthrough: https://drive.google.com/file/d/1iMmNWxnpymCZrK_3kN1b_PtKZ494dlxv/view?usp=drive_link

## �🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Kaggle API configured
# Place kaggle.json in ~/.kaggle/
# Get it from: https://www.kaggle.com/account
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DS_Methodologies

# Install dependencies
pip install -r requirements.txt

# Or use Docker
docker build -t ds-methodologies .
docker run -p 8888:8888 -p 8000:8000 -v ~/.kaggle:/root/.kaggle ds-methodologies
```

### Running Notebooks

Each notebook automatically downloads its dataset on first run via Kaggle API.

```bash
# Launch Jupyter
jupyter notebook

# Navigate to:
# - crisp_dm/CRISP_DM.ipynb
# - semma/SEMMA.ipynb
# - kdd/KDD.ipynb

# Run all cells top-to-bottom
```

### Running on Google Colab

Each methodology includes a Colab-optimized version:

- [CRISP_DM_colab.ipynb](crisp_dm/colab/CRISP_DM_colab.ipynb)
- [SEMMA_colab.ipynb](semma/colab/SEMMA_colab.ipynb)
- [KDD_colab.ipynb](kdd/colab/KDD_colab.ipynb)

**Note**: Upload your `kaggle.json` when prompted in Colab.

## 📁 Repository Structure

```
DS_Methodologies/
├── crisp_dm/                         # CRISP-DM: Walmart Sales
│   ├── CRISP_DM.ipynb                # Single notebook (all phases)
│   ├── colab/CRISP_DM_colab.ipynb    # Colab version
│   ├── prompts/                      # Prompt engineering & critiques
│   │   ├── 00_master_prompt.md
│   │   ├── critic_persona.md
│   │   └── executed/                 # Timestamped prompt logs
│   ├── src/                          # Python modules
│   ├── data/                         # Raw & processed data
│   ├── deployment/                   # FastAPI app
│   ├── reports/                      # Business docs, evaluation
│   └── tests/                        # Unit & integration tests
│
├── semma/                            # SEMMA: Student Performance
│   ├── SEMMA.ipynb                   # Single notebook (S,E,M,M,A)
│   ├── colab/SEMMA_colab.ipynb
│   ├── prompts/
│   ├── sas/                          # SAS implementation
│   ├── src/
│   ├── data/
│   ├── reports/
│   └── tests/
│
├── kdd/                              # KDD: NSL-KDD Intrusion
│   ├── KDD.ipynb                     # Single notebook (Selection→Evaluation)
│   ├── colab/KDD_colab.ipynb
│   ├── prompts/
│   ├── src/
│   ├── data/
│   ├── deployment/
│   ├── reports/
│   └── tests/
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container setup
└── .gitignore                        # Git exclusions
```

## 🎓 Learning Approach

Each notebook follows a **critic-driven development** approach:

1. **Master Prompt**: Defines methodology goals, dataset, and success criteria
2. **Phase Implementation**: Code with production-quality practices
3. **Expert Critique**: World-renowned persona reviews the work
4. **Iteration**: Implement critique recommendations
5. **Documentation**: All prompts and critiques saved to `prompts/executed/`

## 📈 Key Features

### Production-Quality Code
- Modular design with separate `src/` modules
- Comprehensive error handling
- Type hints and docstrings
- Unit and integration tests

### Robust Data Pipeline
- Automatic dataset download via Kaggle API
- Data validation and quality checks
- Reproducible preprocessing pipelines
- Version control for processed data

### Model Development
- Multiple algorithms compared
- Proper cross-validation strategies
- Hyperparameter tuning
- Model interpretability (SHAP)

### Deployment Ready
- FastAPI endpoints
- Docker containerization
- Monitoring and drift detection
- Comprehensive logging

## 🧪 Testing

```bash
# Run tests for specific methodology
pytest crisp_dm/tests/ -v
pytest semma/tests/ -v
pytest kdd/tests/ -v

# Run all tests
pytest -v
```

## 📊 Performance Benchmarks

### CRISP-DM (Walmart Sales)
- **MAE**: Target < 2000 (units)
- **sMAPE**: Target < 15%
- **Beat baseline**: Naive models by 20%+

### SEMMA (Student Performance)
- **Accuracy**: Target > 85%
- **ROC-AUC**: Target > 0.90
- **F1-Score**: Balanced across classes

### KDD (Intrusion Detection)
- **Overall Accuracy**: Target > 95%
- **False Negative Rate**: Minimize for attacks
- **Per-class F1**: All classes > 0.85

## 🔧 Technologies Used

- **Python**: 3.9+
- **ML/DS**: scikit-learn, XGBoost, LightGBM, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretability**: SHAP
- **Tracking**: MLflow
- **Deployment**: FastAPI, Docker
- **Monitoring**: Evidently
- **Testing**: pytest

## 📚 Documentation

Each methodology folder contains:
- `reports/business_understanding.md` - Problem definition and KPIs
- `reports/data_dictionary.md` - Feature descriptions
- `reports/evaluation.md` - Model performance analysis
- `reports/monitoring_plan.md` - Production monitoring strategy (where applicable)

## 🤝 Contributing

This is a portfolio project demonstrating best practices in data science project organization and methodology application.

## 📄 License

MIT License - See individual dataset licenses on Kaggle.

## 🙏 Acknowledgments

- **Datasets**: Kaggle community
- **Methodologies**: CRISP-DM (Chapman et al.), SEMMA (SAS Institute), KDD (Fayyad et al.)
- **Inspiration**: Real-world industry practices

## 📞 Contact

For questions or collaboration opportunities, please reach out via GitHub issues.

---

**Last Updated**: November 2025  
**Status**: Production-Ready Portfolio
