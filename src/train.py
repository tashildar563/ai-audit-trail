import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/diabetes.csv")

# Replace zero values with NaN for medical columns (zeros are impossible)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# ── Features & Target ──────────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

feature_names = X.columns.tolist()

# ── Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale Features ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train Model ────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("=" * 50)
print("        MODEL EVALUATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── Save Model & Scaler ────────────────────────────────────
os.makedirs("model", exist_ok=True)

# Use XGBoost native format instead of pickle (avoids version warnings)
model.save_model("model/xgb_model.json")

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Model saved in native XGBoost format (model/xgb_model.json)")

# ── Save test data for drift monitoring later ──────────────
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
X_test_df["Outcome"] = y_test.values
X_test_df.to_csv("data/reference_data.csv", index=False)
print("✅ Reference data saved to /data/reference_data.csv")