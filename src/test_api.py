# test_api.py - Create this file to test
import requests
import json

# Test single prediction
url = "http://localhost:5000/predict"
data = {
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47,
    "patient_id": "TEST-001"
}

response = requests.post(url, json=data)
print("Single Prediction Response:")
print(json.dumps(response.json(), indent=2))

# Test batch prediction
batch_url = "http://localhost:5000/predict/batch"
batch_data = {
    "patients": [
        {
            "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
            "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627, "Age": 50
        },
        {
            "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66,
            "SkinThickness": 29, "Insulin": 0, "BMI": 26.6,
            "DiabetesPedigreeFunction": 0.351, "Age": 31
        }
    ]
}

batch_response = requests.post(batch_url, json=batch_data)
print("\n\nBatch Prediction Response:")
print(json.dumps(batch_response.json(), indent=2))