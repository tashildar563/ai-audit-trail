import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import os

# ── Load Artifacts ─────────────────────────────────────────
from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model("model/xgb_model.json")
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ── Build SHAP Explainer ───────────────────────────────────
explainer = shap.TreeExplainer(model)

os.makedirs("logs/shap_plots", exist_ok=True)

# ── Explain a Single Prediction ────────────────────────────
def explain_prediction(patient_data: dict, patient_id: str = "UNKNOWN"):
    """
    Returns SHAP values and saves a waterfall plot for the patient.
    """
    input_df = pd.DataFrame([patient_data])
    input_scaled = scaler.transform(input_df[feature_names])
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    shap_values = explainer(input_scaled_df)

    # ── Save Waterfall Plot ────────────────────────────────
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plot_path = f"logs/shap_plots/{patient_id}_shap.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=100)
    plt.close()

    # ── Top Contributing Features ──────────────────────────
    vals = shap_values[0].values
    top_features = sorted(
        zip(feature_names, vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\n📊 SHAP Explanation for {patient_id}:")
    print("-" * 40)
    for feat, val in top_features:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        print(f"  {feat:30s} {direction} risk  (SHAP: {val:+.4f})")
    print(f"\n  Plot saved → {plot_path}")

    return top_features, plot_path


# ── Global Feature Importance ──────────────────────────────
def global_shap_summary():
    """
    Generates a SHAP summary plot using reference/test data.
    """
    ref_df = pd.read_csv("data/reference_data.csv")
    X_ref = ref_df.drop("Outcome", axis=1)[feature_names]

    shap_values = explainer(X_ref)

    plt.figure()
    shap.summary_plot(shap_values, X_ref, show=False)
    path = "logs/shap_plots/global_summary.png"
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"✅ Global SHAP summary saved → {path}")
    return path


# ── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "Pregnancies": 2, "Glucose": 138, "BloodPressure": 62,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 47
    }
    explain_prediction(sample, patient_id="PAT-TEST-001")
    global_shap_summary()