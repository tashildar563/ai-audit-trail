import pandas as pd
import numpy as np
from scipy import stats
import sqlite3
import json
from datetime import datetime
import os

DB_PATH     = "logs/audit_log.db"
REF_PATH    = "data/reference_data.csv"
ALERT_PATH  = "logs/drift_alerts.json"

FEATURE_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age"
]

# ── Load Reference Distribution ────────────────────────────
def load_reference():
    ref = pd.read_csv(REF_PATH)
    # Rename to match audit log column names
    ref.columns = [c.lower().replace("diabetespedigreefunction", "diabetes_pedigree")
                   .replace("bloodpressure", "blood_pressure")
                   .replace("skinthickness", "skin_thickness")
                   for c in ref.columns]
    return ref[FEATURE_COLS]


# ── Load Recent Predictions from Audit Log ─────────────────
def load_recent_predictions(limit=50):
    if not os.path.exists(DB_PATH):
        print("⚠️  No audit log found. Run some predictions first.")
        return None
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT pregnancies, glucose, blood_pressure, skin_thickness,
               insulin, bmi, diabetes_pedigree, age
        FROM audit_log
        ORDER BY id DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


# ── KS Test for Drift ──────────────────────────────────────
def detect_drift(ref_df, current_df, threshold=0.05):
    """
    Runs Kolmogorov-Smirnov test per feature.
    p-value < threshold means drift detected.
    """
    results  = {}
    drifted  = []

    for col in FEATURE_COLS:
        if col not in current_df.columns:
            continue
        stat, p_value = stats.ks_2samp(
            ref_df[col].dropna(),
            current_df[col].dropna()
        )
        is_drift = p_value < threshold
        results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value":      round(float(p_value), 4),
            "drift":        is_drift
        }
        if is_drift:
            drifted.append(col)

    return results, drifted


# ── Save Alert ─────────────────────────────────────────────
def save_alert(drifted_features, results):
    os.makedirs("logs", exist_ok=True)
    alerts = []
    if os.path.exists(ALERT_PATH):
        try:
            with open(ALERT_PATH) as f:
                content = f.read().strip()
                if content:
                    alerts = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            alerts = []

    alerts.append({
        "timestamp":        datetime.now().isoformat(),
        "drifted_features": drifted_features,
        "details":          results,
        "severity":         "HIGH" if len(drifted_features) >= 3 else "MEDIUM"
                            if len(drifted_features) >= 1 else "OK"
    })

    with open(ALERT_PATH, "w") as f:
        json.dump(alerts, f, indent=2)


# ── Run Full Drift Check ───────────────────────────────────
def run_drift_check():
    print("\n🔍 Running Drift Detection...")
    print("=" * 50)

    ref_df     = load_reference()
    current_df = load_recent_predictions()

    if current_df is None or len(current_df) < 5:
        print("⚠️  Not enough predictions logged yet (need at least 5).")
        print("   Tip: Run predict.py a few more times with different inputs.")
        return

    results, drifted = detect_drift(ref_df, current_df)

    # ── Print Report ───────────────────────────────────────
    print(f"\n{'Feature':<30} {'KS Stat':>10} {'P-Value':>10} {'Status':>10}")
    print("-" * 65)
    for feat, info in results.items():
        status = "🔴 DRIFT" if info["drift"] else "🟢 OK"
        print(f"{feat:<30} {info['ks_statistic']:>10.4f} "
              f"{info['p_value']:>10.4f} {status:>10}")

    print("\n" + "=" * 50)
    if drifted:
        print(f"⚠️  ALERT: Drift detected in → {', '.join(drifted)}")
        print(f"   Severity: {'HIGH' if len(drifted) >= 3 else 'MEDIUM'}")
        print("   Recommendation: Consider retraining the model.")
    else:
        print("✅ No drift detected. Model inputs look stable.")

    save_alert(drifted, results)
    print(f"\n📁 Alert log saved → {ALERT_PATH}")
    return results, drifted


if __name__ == "__main__":
    run_drift_check()