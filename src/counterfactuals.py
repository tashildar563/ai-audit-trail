import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# Load artifacts
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load champion model
model = XGBClassifier()
model.load_model("model/versions/xgboost_v2.0.json")

def generate_counterfactuals(patient_data, target_outcome="No Diabetes", max_changes=3):
    """
    Generate counterfactual explanations:
    "What would need to change for a different outcome?"
    
    Args:
        patient_data: dict of patient features
        target_outcome: desired prediction outcome
        max_changes: maximum features to modify
    
    Returns:
        List of counterfactual scenarios
    """
    
    # Current prediction
    input_df = pd.DataFrame([patient_data])
    input_scaled = scaler.transform(input_df[feature_names])
    current_pred = model.predict(input_scaled)[0]
    current_label = "Diabetes" if current_pred == 1 else "No Diabetes"
    
    if current_label == target_outcome:
        return {
            "message": f"Patient already predicted as {target_outcome}",
            "counterfactuals": []
        }
    
    target_class = 0 if target_outcome == "No Diabetes" else 1
    
    # Modifiable features and their reasonable ranges
    feature_ranges = {
        "Glucose": (70, 200),
        "BMI": (18, 45),
        "Age": (patient_data["Age"], patient_data["Age"]),  # Can't change age
        "BloodPressure": (60, 100),
        "Insulin": (0, 300),
        "Pregnancies": (patient_data["Pregnancies"], patient_data["Pregnancies"]),  # Can't change
        "SkinThickness": (10, 50),
        "DiabetesPedigreeFunction": (0.1, 1.5)
    }
    
    counterfactuals = []
    
    # Strategy 1: Reduce high-risk features
    if target_outcome == "No Diabetes":
        scenarios = [
            {"name": "Glucose Reduction", "feature": "Glucose", "change": -20},
            {"name": "BMI Reduction", "feature": "BMI", "change": -3.0},
            {"name": "Combined Lifestyle", "features": ["Glucose", "BMI"], "changes": [-15, -2.5]}
        ]
    else:
        scenarios = [
            {"name": "Glucose Increase", "feature": "Glucose", "change": +20},
            {"name": "BMI Increase", "feature": "BMI", "change": +3.0}
        ]
    
    for scenario in scenarios:
        modified = patient_data.copy()
        
        if "feature" in scenario:
            # Single feature change
            feat = scenario["feature"]
            modified[feat] = patient_data[feat] + scenario["change"]
            
            # Clip to valid range
            min_val, max_val = feature_ranges[feat]
            modified[feat] = np.clip(modified[feat], min_val, max_val)
        
        elif "features" in scenario:
            # Multiple feature changes
            for feat, change in zip(scenario["features"], scenario["changes"]):
                modified[feat] = patient_data[feat] + change
                min_val, max_val = feature_ranges[feat]
                modified[feat] = np.clip(modified[feat], min_val, max_val)
        
        # Test modified scenario
        mod_df = pd.DataFrame([modified])
        mod_scaled = scaler.transform(mod_df[feature_names])
        new_pred = model.predict(mod_scaled)[0]
        new_prob = model.predict_proba(mod_scaled)[0]
        new_label = "Diabetes" if new_pred == 1 else "No Diabetes"
        
        if new_label == target_outcome:
            changes_made = []
            for feat in feature_names:
                if abs(modified[feat] - patient_data[feat]) > 0.01:
                    changes_made.append({
                        "feature": feat,
                        "original": round(patient_data[feat], 2),
                        "modified": round(modified[feat], 2),
                        "change": round(modified[feat] - patient_data[feat], 2)
                    })
            
            counterfactuals.append({
                "scenario": scenario["name"],
                "outcome": new_label,
                "confidence": round(float(new_prob[new_pred]), 4),
                "changes": changes_made,
                "feasibility": "HIGH" if len(changes_made) <= 2 else "MEDIUM"
            })
    
    return {
        "current_prediction": current_label,
        "target_outcome": target_outcome,
        "counterfactuals": counterfactuals[:max_changes]
    }


if __name__ == "__main__":
    # Test
    sample = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    result = generate_counterfactuals(sample, target_outcome="No Diabetes")
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL EXPLANATIONS")
    print("="*60)
    print(f"\nCurrent: {result['current_prediction']}")
    print(f"Target:  {result['target_outcome']}")
    print(f"\nFound {len(result['counterfactuals'])} counterfactual scenarios:\n")
    
    for i, cf in enumerate(result['counterfactuals'], 1):
        print(f"{i}. {cf['scenario']} (Feasibility: {cf['feasibility']})")
        print(f"   → Outcome: {cf['outcome']} (confidence: {cf['confidence']:.1%})")
        print(f"   Changes needed:")
        for change in cf['changes']:
            print(f"     • {change['feature']}: {change['original']} → {change['modified']} ({change['change']:+.1f})")
        print()