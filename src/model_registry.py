import os
import json
import pickle
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

REGISTRY_PATH = "model/registry.json"
MODEL_DIR = "model/versions"

os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# MODEL REGISTRY MANAGEMENT
# ══════════════════════════════════════════════════════════

def load_registry():
    """Load the model registry (tracks all trained models)"""
    if not os.path.exists(REGISTRY_PATH):
        return {"models": [], "champion": None}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry):
    """Save the model registry"""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    model_name: str,
    model_type: str,
    version: str,
    metrics: dict,
    is_champion: bool = False
):
    """Register a new model in the registry"""
    registry = load_registry()
    
    model_entry = {
        "model_name": model_name,
        "model_type": model_type,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "status": "champion" if is_champion else "challenger",
        "model_path": f"{MODEL_DIR}/{model_name}_{version}.json"
        if model_type == "xgboost" else f"{MODEL_DIR}/{model_name}_{version}.pkl"
    }
    
    registry["models"].append(model_entry)
    
    if is_champion:
        registry["champion"] = model_name
    
    save_registry(registry)
    print(f"✅ Registered {model_name} v{version} as {model_entry['status']}")
    return model_entry


def get_champion_model():
    """Get the current champion model"""
    registry = load_registry()
    champion_name = registry.get("champion")
    
    if not champion_name:
        return None
    
    for model in registry["models"]:
        if model["model_name"] == champion_name and model["status"] == "champion":
            return model
    return None


def set_champion(model_name: str):
    """Promote a model to champion"""
    registry = load_registry()
    
    # Demote current champion
    for model in registry["models"]:
        if model["status"] == "champion":
            model["status"] = "challenger"
    
    # Promote new champion
    for model in registry["models"]:
        if model["model_name"] == model_name:
            model["status"] = "champion"
            registry["champion"] = model_name
            break
    
    save_registry(registry)
    print(f"👑 {model_name} is now the champion model!")


def get_all_models():
    """Get all registered models"""
    registry = load_registry()
    return pd.DataFrame(registry["models"])


# ══════════════════════════════════════════════════════════
# MULTI-MODEL TRAINING
# ══════════════════════════════════════════════════════════

def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost, Random Forest, and Logistic Regression"""
    
    results = {}
    
    # ── Model 1: XGBoost ───────────────────────────────────
    print("\n🚀 Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    
    xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")
    xgb.save_model(f"{MODEL_DIR}/xgboost_v2.0.json")
    
    register_model(
        model_name="xgboost",
        model_type="xgboost",
        version="v2.0",
        metrics=xgb_metrics,
        is_champion=True  # Set as initial champion
    )
    results["xgboost"] = {"model": xgb, "metrics": xgb_metrics}
    
    # ── Model 2: Random Forest ─────────────────────────────
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    rf_metrics = evaluate_model(rf, X_test, y_test, "RandomForest")
    
    with open(f"{MODEL_DIR}/randomforest_v1.0.pkl", "wb") as f:
        pickle.dump(rf, f)
    
    register_model(
        model_name="randomforest",
        model_type="sklearn",
        version="v1.0",
        metrics=rf_metrics,
        is_champion=False
    )
    results["randomforest"] = {"model": rf, "metrics": rf_metrics}
    
    # ── Model 3: Logistic Regression ───────────────────────
    print("\n📊 Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    
    lr_metrics = evaluate_model(lr, X_test, y_test, "LogisticRegression")
    
    with open(f"{MODEL_DIR}/logistic_v1.0.pkl", "wb") as f:
        pickle.dump(lr, f)
    
    register_model(
        model_name="logistic",
        model_type="sklearn",
        version="v1.0",
        metrics=lr_metrics,
        is_champion=False
    )
    results["logistic"] = {"model": lr, "metrics": lr_metrics}
    
    return results


def evaluate_model(model, X_test, y_test, model_name):
    """Calculate comprehensive metrics for a model"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    print(f"\n📊 {model_name} Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall:    {metrics['recall']:.2%}")
    print(f"   F1 Score:  {metrics['f1_score']:.2%}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


# ══════════════════════════════════════════════════════════
# LOAD MODEL BY NAME
# ══════════════════════════════════════════════════════════

def load_model_by_name(model_name: str):
    """Load a specific model from registry"""
    registry = load_registry()
    
    model_entry = None
    for model in registry["models"]:
        if model["model_name"] == model_name:
            model_entry = model
            break
    
    if not model_entry:
        raise ValueError(f"Model {model_name} not found in registry")
    
    model_path = model_entry["model_path"]
    
    if model_entry["model_type"] == "xgboost":
        model = XGBClassifier()
        model.load_model(model_path)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    return model, model_entry