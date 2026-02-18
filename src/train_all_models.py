import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_registry import train_all_models

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/diabetes.csv")

# Clean zeros
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = X.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train all models
print("\n" + "="*60)
print("  TRAINING MULTI-MODEL GOVERNANCE SYSTEM")
print("="*60)

results = train_all_models(
    X_train_scaled, X_test_scaled,
    y_train, y_test,
    feature_names
)

# Save scaler (same for all models)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n" + "="*60)
print("✅ All models trained and registered!")
print("="*60)

# Display comparison
print("\n📊 MODEL COMPARISON:")
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-" * 80)

for name, data in results.items():
    m = data["metrics"]
    print(f"{name:<20} {m['accuracy']:<12.2%} {m['precision']:<12.2%} "
          f"{m['recall']:<12.2%} {m['f1_score']:<12.2%} {m['roc_auc']:<12.4f}")