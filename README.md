# AI Decision Audit Trail System

> **Production-grade AI governance platform** with multi-model management, fairness analysis, drift monitoring, and comprehensive explainability for medical diagnosis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Project Overview

A comprehensive **AI governance and monitoring platform** designed for healthcare ML systems. This project demonstrates enterprise-grade MLOps practices including model versioning, fairness auditing, drift detection, and full prediction auditability.

### **Key Features**

- 🤖 **Multi-Model Framework** — XGBoost, Random Forest, and Logistic Regression with Champion/Challenger A/B testing
- ⚖️ **Fairness & Bias Detection** — Demographic parity analysis, disparate impact (80% rule), equal opportunity metrics
- 📊 **Production Monitoring** — Real-time performance tracking, data quality validation, automated degradation alerts
- 🔍 **Explainability** — SHAP-based explanations for every prediction with feature contribution analysis
- 📉 **Drift Detection** — Statistical drift monitoring using Kolmogorov-Smirnov tests with automated alerts
- 🗄️ **Complete Audit Trail** — Every prediction logged to SQLite with full provenance tracking
- 🌐 **REST API** — Production-ready Flask API with batch processing endpoints
- 📱 **Interactive Dashboard** — Streamlit-based UI with 9 comprehensive monitoring pages

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │ Streamlit Dashboard│        │   REST API        │        │
│  │  (9 pages)        │          │  (Flask)          │        │
│  └──────────────────┘          └──────────────────┘        │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
┌────────────▼────────────────────────────▼───────────────────┐
│                   Application Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Predict  │  │ Explain  │  │  Drift   │  │ Fairness │   │
│  │  Engine  │  │  (SHAP)  │  │ Monitor  │  │ Analyzer │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                     Model Layer                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Model Registry & Versioning                 │  │
│  ├──────────────┬──────────────┬──────────────────────┤  │
│  │   XGBoost    │ Random Forest │ Logistic Regression  │  │
│  │   (Champion) │  (Challenger) │    (Challenger)      │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                     Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Audit Log    │  │ Performance  │  │  Fairness    │     │
│  │  (SQLite)    │  │    Logs      │  │   Reports    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11
- pip package manager
- 4GB+ RAM
- 1GB free disk space

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-audit-trail.git
cd ai-audit-trail

# Install dependencies
pip install -r requirements.txt

# Train initial models
python src/train_all_models.py

# Generate sample predictions
python src/generate_samples.py

# Launch dashboard
streamlit run app/dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Quick API Start
```bash
# Start the API server (in a separate terminal)
python src/api.py

# Test the API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47
  }'
```

---

## 📊 Dashboard Features

### 1. 🏠 Home — Overview & Analytics
- Real-time metrics (total predictions, diabetes detection rate)
- Interactive prediction timeline
- Confidence distribution analysis
- Risk stratification by age groups
- Feature correlation heatmap

### 2. 🔮 New Prediction — Single Patient Analysis
- Interactive input form with sliders
- Real-time prediction with confidence scores
- SHAP waterfall plot for explainability
- Top 5 contributing features with impact direction

### 3. 📋 Audit Log — Complete Traceability
- Searchable log of all predictions
- Filter by diagnosis, confidence, patient ID
- Full export to CSV
- Timestamp tracking for compliance

### 4. 📉 Drift Monitor — Data Distribution Tracking
- Kolmogorov-Smirnov statistical tests per feature
- Visual distribution comparisons (reference vs. current)
- Automated alert generation
- Drift history timeline

### 5. 📊 Model Report — Performance Overview
- Model architecture details
- Performance metrics (accuracy, ROC-AUC, F1)
- Global SHAP feature importance
- Individual prediction explanation gallery

### 6. 🏆 Model Comparison — Multi-Model Governance
- Side-by-side performance comparison (XGBoost, RF, Logistic Regression)
- Radar charts for multi-metric visualization
- Confusion matrix comparison
- Champion/Challenger promotion workflow

### 7. ⚖️ Fairness & Bias — Responsible AI
- Demographic subgroup analysis (age groups)
- Disparate Impact Ratio (80% rule compliance)
- Equal Opportunity & Equalized Odds metrics
- Bias mitigation recommendations
- Downloadable fairness audit reports

### 8. 🏭 Production Monitor — Operational Excellence
- Performance tracking over time (daily metrics)
- Data quality monitoring (missing values, outliers, range validation)
- System health dashboard (prediction volume, uptime)
- Automated retraining triggers
- Production alert log

### 9. 📤 Batch Processing — Scalable Screening
- CSV upload for bulk predictions (up to 1000 patients)
- Risk distribution visualization
- Confidence analysis
- Downloadable batch results
- Template generator

---

## 🔧 Technical Stack

| Layer | Technology |
|-------|-----------|
| **ML Models** | XGBoost, scikit-learn (Random Forest, Logistic Regression) |
| **Explainability** | SHAP (TreeExplainer) |
| **Drift Detection** | Evidently AI, scipy (KS tests) |
| **API** | Flask, Flask-CORS |
| **Frontend** | Streamlit, Plotly |
| **Database** | SQLite |
| **Data Processing** | pandas, numpy |
| **Visualization** | Plotly, matplotlib, seaborn |

---

## 📁 Project Structure
```
ai-audit-trail/
│
├── data/
│   ├── diabetes.csv              # Training dataset
│   └── reference_data.csv        # Drift detection baseline
│
├── model/
│   ├── versions/                 # Model version storage
│   │   ├── xgboost_v2.0.json
│   │   ├── randomforest_v1.0.pkl
│   │   └── logistic_v1.0.pkl
│   ├── registry.json             # Model metadata registry
│   ├── scaler.pkl                # Feature scaler
│   └── feature_names.pkl         # Feature schema
│
├── logs/
│   ├── audit_log.db              # SQLite prediction log
│   ├── drift_alerts.json         # Drift detection alerts
│   ├── fairness_report.json      # Fairness analysis
│   ├── data_quality_log.json     # Data quality metrics
│   ├── production_alerts.json    # System alerts
│   └── shap_plots/               # SHAP visualizations
│
├── src/
│   ├── train.py                  # Single model training
│   ├── train_all_models.py       # Multi-model training
│   ├── predict.py                # Prediction + logging
│   ├── explainer.py              # SHAP explanations
│   ├── drift_monitor.py          # Drift detection
│   ├── fairness_analysis.py      # Bias detection
│   ├── production_monitor.py     # Performance tracking
│   ├── model_registry.py         # Model versioning
│   ├── counterfactuals.py        # Counterfactual explanations
│   ├── api.py                    # REST API server
│   └── generate_samples.py       # Sample data generator
│
├── app/
│   └── dashboard.py              # Streamlit dashboard (9 pages)
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

---

## 🔍 Key Metrics & Performance

### Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost (Champion)** | 77.3% | 73.2% | 67.1% | 70.0% | 0.8421 |
| Random Forest | 75.8% | 71.5% | 65.3% | 68.2% | 0.8293 |
| Logistic Regression | 76.6% | 72.8% | 64.7% | 68.5% | 0.8156 |

### Fairness Metrics

- **Disparate Impact Ratio**: 0.92 (passes 80% rule ✅)
- **Max TPR Disparity**: 0.08 (below 0.10 threshold ✅)
- **Equalized Odds**: 0.06 (acceptable ✅)

### System Performance

- **Average Prediction Latency**: <50ms
- **API Throughput**: ~200 predictions/second
- **Dashboard Load Time**: <2 seconds

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- ✅ **MLOps**: Model versioning, A/B testing, automated retraining
- ✅ **Responsible AI**: Fairness auditing, bias detection, explainability
- ✅ **Production ML**: Drift monitoring, performance tracking, data quality validation
- ✅ **Software Engineering**: REST API design, database modeling, clean architecture
- ✅ **Data Visualization**: Interactive dashboards, statistical charts
- ✅ **Healthcare AI**: Medical domain knowledge, HIPAA-aware design patterns

---

## 📖 Usage Examples

### Python API
```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', json={
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47,
    "patient_id": "PAT-12345"
})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Risk Level: {result['risk_level']}")
```

### Command Line Monitoring
```bash
# Run performance monitoring
python src/production_monitor.py

# Run fairness analysis
python src/fairness_analysis.py

# Run drift detection
python src/drift_monitor.py

# Generate counterfactuals
python src/counterfactuals.py
```

---

## 🚧 Future Enhancements

- [ ] PostgreSQL migration for production scale
- [ ] Docker containerization with docker-compose
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Kubernetes deployment manifests
- [ ] Real-time streaming predictions (Kafka integration)
- [ ] Advanced fairness interventions (reweighting, threshold optimization)
- [ ] Multi-cloud deployment (AWS SageMaker, Azure ML)
- [ ] OAuth2 authentication for API
- [ ] Grafana dashboard integration
- [ ] Model serving with TensorFlow Serving / Triton

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Pima Indians Diabetes Database (UCI ML Repository)
- **Frameworks**: scikit-learn, XGBoost, SHAP, Streamlit, Flask
- **Inspiration**: Industry best practices in ML governance and responsible AI

---

## 📧 Contact

**Rakesh Narayan Tashildar**  
📧 tashildar563@gmail.com  
🔗 [LinkedIn](https://linkedin.com/)  
🐙 [GitHub](https://github.com/tashildar563)

---

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

**Built with ❤️ for responsible AI and production ML excellence**