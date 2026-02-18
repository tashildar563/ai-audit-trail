# run this as: python src/generate_samples.py
import sys
sys.path.append(".")
from src.predict import predict_and_log
import random

samples = [
    {"Pregnancies": 6,  "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0,   "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50},
    {"Pregnancies": 1,  "Glucose": 85,  "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0,   "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31},
    {"Pregnancies": 8,  "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0,  "Insulin": 0,   "BMI": 23.3, "DiabetesPedigreeFunction": 0.672, "Age": 32},
    {"Pregnancies": 1,  "Glucose": 89,  "BloodPressure": 66, "SkinThickness": 23, "Insulin": 94,  "BMI": 28.1, "DiabetesPedigreeFunction": 0.167, "Age": 21},
    {"Pregnancies": 0,  "Glucose": 137, "BloodPressure": 40, "SkinThickness": 35, "Insulin": 168, "BMI": 43.1, "DiabetesPedigreeFunction": 2.288, "Age": 33},
    {"Pregnancies": 5,  "Glucose": 116, "BloodPressure": 74, "SkinThickness": 0,  "Insulin": 0,   "BMI": 25.6, "DiabetesPedigreeFunction": 0.201, "Age": 30},
    {"Pregnancies": 3,  "Glucose": 78,  "BloodPressure": 50, "SkinThickness": 32, "Insulin": 88,  "BMI": 31.0, "DiabetesPedigreeFunction": 0.248, "Age": 26},
    {"Pregnancies": 10, "Glucose": 168, "BloodPressure": 74, "SkinThickness": 0,  "Insulin": 0,   "BMI": 38.0, "DiabetesPedigreeFunction": 0.537, "Age": 34},
    {"Pregnancies": 2,  "Glucose": 99,  "BloodPressure": 52, "SkinThickness": 15, "Insulin": 94,  "BMI": 24.6, "DiabetesPedigreeFunction": 0.637, "Age": 40},
    {"Pregnancies": 7,  "Glucose": 196, "BloodPressure": 90, "SkinThickness": 0,  "Insulin": 0,   "BMI": 39.8, "DiabetesPedigreeFunction": 0.451, "Age": 41},
]

for i, s in enumerate(samples):
    predict_and_log(s, patient_id=f"PAT-{i+1:03d}")

print("\n✅ 10 sample predictions logged!")