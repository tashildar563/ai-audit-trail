# Quick Start Guide

## ⚡ 5-Minute Setup
```bash
# 1. Clone & install
git clone https://github.com/yourusername/ai-audit-trail.git
cd ai-audit-trail
pip install -r requirements.txt

# 2. Train models (takes ~30 seconds)
python src/train_all_models.py

# 3. Generate sample data
python src/generate_samples.py

# 4. Launch dashboard
streamlit run app/dashboard.py
```

Open browser to http://localhost:8501

## 🎯 What to Try First

1. **Home Page** — See analytics overview
2. **New Prediction** — Make a prediction and see SHAP explanation
3. **Model Comparison** — Compare 3 trained models
4. **Fairness & Bias** — Run fairness analysis
5. **Production Monitor** — Check system health

## 🌐 API Usage
```bash
# Terminal 1: Start API
python src/api.py

# Terminal 2: Test
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## 📊 Sample Data

The system uses the Pima Indians Diabetes dataset:
- 768 patients
- 8 features (glucose, BMI, age, etc.)
- Binary outcome (diabetes / no diabetes)

## ❓ Troubleshooting

**Dashboard won't start**: Check Python version (3.8-3.11)
**API errors**: Ensure models are trained first
**Empty dashboard**: Run `python src/generate_samples.py`