import sqlite3
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ── Load Model Artifacts ───────────────────────────────────
from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model("model/xgb_model.json")
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ── Initialize Database ────────────────────────────────────
os.makedirs("logs", exist_ok=True)
DB_PATH = "logs/audit_log.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            patient_id       TEXT,
            pregnancies      REAL,
            glucose          REAL,
            blood_pressure   REAL,
            skin_thickness   REAL,
            insulin          REAL,
            bmi              REAL,
            diabetes_pedigree REAL,
            age              REAL,
            prediction       INTEGER,
            prediction_label TEXT,
            confidence       REAL,
            model_version    TEXT
        )
    """)
    conn.commit()
    conn.close()

# ── Make & Log Prediction ──────────────────────────────────
def predict_and_log(patient_data: dict, patient_id: str = None):
    """
    patient_data: dict with keys matching feature_names
    Returns: prediction label, confidence score
    """
    init_db()

    # Prepare input
    input_df = pd.DataFrame([patient_data])
    input_scaled = scaler.transform(input_df[feature_names])

    # Predict
    prediction = int(model.predict(input_scaled)[0])
    confidence = float(model.predict_proba(input_scaled)[0][prediction])
    label = "Diabetes" if prediction == 1 else "No Diabetes"
    patient_id = patient_id or f"PAT-{datetime.now().strftime('%H%M%S%f')[:10]}"

    # Log to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_log (
            timestamp, patient_id,
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age,
            prediction, prediction_label, confidence, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        patient_id,
        patient_data.get("Pregnancies"),
        patient_data.get("Glucose"),
        patient_data.get("BloodPressure"),
        patient_data.get("SkinThickness"),
        patient_data.get("Insulin"),
        patient_data.get("BMI"),
        patient_data.get("DiabetesPedigreeFunction"),
        patient_data.get("Age"),
        prediction,
        label,
        round(confidence, 4),
        "xgb_v1.0"
    ))
    conn.commit()
    conn.close()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Patient {patient_id} → {label} (confidence: {confidence:.2%})")
    return label, confidence

# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "Pregnancies": 2, "Glucose": 138, "BloodPressure": 62,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 47
    }
    label, conf = predict_and_log(sample, patient_id="PAT-TEST-001")
    print(f"\nResult: {label} with {conf:.2%} confidence")