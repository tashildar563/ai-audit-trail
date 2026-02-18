from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.predict import predict_and_log
from src.explainer import explain_prediction
from src.model_registry import get_champion_model, get_all_models, load_model_by_name

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════

def load_artifacts():
    """Load scaler and feature names"""
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return scaler, feature_names

scaler, feature_names = load_artifacts()

# ══════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        "name": "AI Audit Trail API",
        "version": "1.0.0",
        "description": "Production-grade diabetes prediction API with full audit trail",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Make a single prediction",
                "example": {
                    "Pregnancies": 2,
                    "Glucose": 138,
                    "BloodPressure": 62,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 47
                }
            },
            "/predict/batch": {
                "method": "POST",
                "description": "Make predictions for multiple patients",
                "example": {
                    "patients": [
                        {"Pregnancies": 2, "Glucose": 138, "...": "..."},
                        {"Pregnancies": 1, "Glucose": 85, "...": "..."}
                    ]
                }
            },
            "/models": {
                "method": "GET",
                "description": "List all available models"
            },
            "/models/champion": {
                "method": "GET",
                "description": "Get current champion model info"
            },
            "/health": {
                "method": "GET",
                "description": "API health check"
            }
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Audit Trail API"
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Example request:
    {
        "Pregnancies": 2,
        "Glucose": 138,
        "BloodPressure": 62,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 47,
        "patient_id": "PAT-001" (optional)
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        for field in feature_names:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}"
                }), 400
        
        patient_id = data.get("patient_id", f"API-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        # Extract patient data
        patient_data = {k: data[k] for k in feature_names}
        
        # Make prediction (logs automatically)
        label, confidence = predict_and_log(patient_data, patient_id)
        
        # Get explanation
        top_features, plot_path = explain_prediction(patient_data, patient_id)
        
        # Format response
        response = {
            "patient_id": patient_id,
            "prediction": label,
            "confidence": round(confidence, 4),
            "risk_level": "HIGH" if label == "Diabetes" and confidence > 0.8
                         else "MEDIUM" if label == "Diabetes"
                         else "LOW",
            "top_contributing_features": [
                {
                    "feature": feat,
                    "shap_value": round(val, 4),
                    "impact": "increases_risk" if val > 0 else "decreases_risk"
                }
                for feat, val in top_features[:5]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Example request:
    {
        "patients": [
            {"Pregnancies": 2, "Glucose": 138, ...},
            {"Pregnancies": 1, "Glucose": 85, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if "patients" not in data or not isinstance(data["patients"], list):
            return jsonify({
                "error": "Request must contain 'patients' array"
            }), 400
        
        patients = data["patients"]
        
        if len(patients) > 1000:
            return jsonify({
                "error": "Batch size limit: 1000 patients"
            }), 400
        
        results = []
        
        for i, patient in enumerate(patients):
            patient_id = patient.get("patient_id", f"BATCH-{i+1}")
            
            # Validate fields
            missing = [f for f in feature_names if f not in patient]
            if missing:
                results.append({
                    "patient_id": patient_id,
                    "error": f"Missing fields: {missing}"
                })
                continue
            
            # Extract and predict
            patient_data = {k: patient[k] for k in feature_names}
            label, confidence = predict_and_log(patient_data, patient_id)
            
            results.append({
                "patient_id": patient_id,
                "prediction": label,
                "confidence": round(confidence, 4),
                "risk_level": "HIGH" if label == "Diabetes" and confidence > 0.8
                             else "MEDIUM" if label == "Diabetes"
                             else "LOW"
            })
        
        # Summary statistics
        total = len(results)
        successful = len([r for r in results if "error" not in r])
        diabetes_detected = len([r for r in results if r.get("prediction") == "Diabetes"])
        high_risk = len([r for r in results if r.get("risk_level") == "HIGH"])
        
        return jsonify({
            "summary": {
                "total_patients": total,
                "successful_predictions": successful,
                "diabetes_detected": diabetes_detected,
                "high_risk_cases": high_risk
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List all registered models"""
    try:
        models_df = get_all_models()
        
        models = []
        for _, row in models_df.iterrows():
            models.append({
                "model_name": row["model_name"],
                "version": row["version"],
                "status": row["status"],
                "trained_at": row["timestamp"],
                "metrics": row["metrics"]
            })
        
        return jsonify({
            "models": models,
            "count": len(models)
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/models/champion', methods=['GET'])
def get_champion():
    """Get current champion model"""
    try:
        champion = get_champion_model()
        
        if not champion:
            return jsonify({
                "error": "No champion model set"
            }), 404
        
        return jsonify({
            "champion": {
                "model_name": champion["model_name"],
                "version": champion["version"],
                "trained_at": champion["timestamp"],
                "metrics": champion["metrics"]
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ══════════════════════════════════════════════════════════
# RUN SERVER
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AI AUDIT TRAIL API SERVER")
    print("="*60)
    print("\n🚀 Starting API server on http://localhost:5000")
    print("\n📚 API Documentation: http://localhost:5000/")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)