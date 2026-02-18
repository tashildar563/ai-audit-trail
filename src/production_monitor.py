import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = "logs/audit_log.db"
PERFORMANCE_LOG_PATH = "logs/performance_log.json"
DATA_QUALITY_LOG_PATH = "logs/data_quality_log.json"
ALERTS_LOG_PATH = "logs/production_alerts.json"

# ══════════════════════════════════════════════════════════
# PERFORMANCE TRACKING
# ══════════════════════════════════════════════════════════

def calculate_rolling_performance(window_hours=24):
    """
    Calculate model performance metrics over rolling time windows.
    In production, you'd compare predictions to ground truth labels.
    For demo, we simulate ground truth with realistic error rates.
    """
    
    if not os.path.exists(DB_PATH):
        return {"error": "No audit log found"}
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT id, timestamp, prediction, confidence,
               glucose, bmi, age, pregnancies
        FROM audit_log
        ORDER BY timestamp DESC
    """, conn)
    conn.close()
    
    if len(df) < 5:
        return {"error": "Need at least 5 predictions for performance tracking"}
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Simulate ground truth (in production, you'd join with actual outcomes)
    np.random.seed(42)
    df['actual'] = df['prediction'].copy()
    
    # Simulate realistic errors (confidence-based)
    for idx in df.index:
        conf = df.loc[idx, 'confidence']
        error_prob = (1 - conf) * 0.3  # Lower confidence = higher error chance
        if np.random.random() < error_prob:
            df.loc[idx, 'actual'] = 1 - df.loc[idx, 'actual']
    
    # Calculate metrics
    df['correct'] = (df['prediction'] == df['actual']).astype(int)
    df['true_positive'] = ((df['prediction'] == 1) & (df['actual'] == 1)).astype(int)
    df['false_positive'] = ((df['prediction'] == 1) & (df['actual'] == 0)).astype(int)
    df['true_negative'] = ((df['prediction'] == 0) & (df['actual'] == 0)).astype(int)
    df['false_negative'] = ((df['prediction'] == 0) & (df['actual'] == 1)).astype(int)
    
    # Time-based aggregation
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.floor('H')
    
    # Daily metrics
    daily_metrics = []
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        
        tp = day_data['true_positive'].sum()
        fp = day_data['false_positive'].sum()
        tn = day_data['true_negative'].sum()
        fn = day_data['false_negative'].sum()
        
        total = len(day_data)
        accuracy = day_data['correct'].mean() if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        daily_metrics.append({
            "date": str(date),
            "predictions": int(total),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "avg_confidence": round(float(day_data['confidence'].mean()), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        })
    
    # Overall current metrics (last 7 days)
    recent = df.head(min(len(df), 100))  # Last 100 predictions
    
    current_metrics = {
        "accuracy": round(float(recent['correct'].mean()), 4),
        "avg_confidence": round(float(recent['confidence'].mean()), 4),
        "predictions_last_24h": int(len(df[df['timestamp'] > datetime.now() - timedelta(hours=24)])),
        "total_predictions": int(len(df))
    }
    
    return {
        "daily_metrics": daily_metrics,
        "current_metrics": current_metrics,
        "timestamp": datetime.now().isoformat()
    }


def detect_performance_degradation(baseline_accuracy=0.77, threshold=0.05):
    """
    Detect if model performance has degraded significantly.
    Returns alerts if accuracy drops below baseline - threshold.
    """
    
    perf_data = calculate_rolling_performance()
    
    if "error" in perf_data:
        return {"degradation_detected": False, "reason": perf_data["error"]}
    
    current_acc = perf_data["current_metrics"]["accuracy"]
    degradation_threshold = baseline_accuracy - threshold
    
    is_degraded = current_acc < degradation_threshold
    
    result = {
        "degradation_detected": is_degraded,
        "current_accuracy": current_acc,
        "baseline_accuracy": baseline_accuracy,
        "threshold": degradation_threshold,
        "drop_amount": round(baseline_accuracy - current_acc, 4),
        "severity": "CRITICAL" if current_acc < (baseline_accuracy - 0.10) 
                   else "HIGH" if is_degraded else "OK"
    }
    
    if is_degraded:
        log_alert("PERFORMANCE_DEGRADATION", result)
    
    return result


# ══════════════════════════════════════════════════════════
# DATA QUALITY MONITORING
# ══════════════════════════════════════════════════════════

def monitor_data_quality():
    """
    Monitor incoming data quality:
    - Missing values
    - Outliers (statistical)
    - Schema validation
    - Distribution shifts
    """
    
    if not os.path.exists(DB_PATH):
        return {"error": "No audit log found"}
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT pregnancies, glucose, blood_pressure, skin_thickness,
               insulin, bmi, diabetes_pedigree, age, timestamp
        FROM audit_log
        ORDER BY timestamp DESC
        LIMIT 100
    """, conn)
    conn.close()
    
    if len(df) < 5:
        return {"error": "Need at least 5 predictions for data quality monitoring"}
    
    feature_cols = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                    'insulin', 'bmi', 'diabetes_pedigree', 'age']
    
    quality_report = {
        "timestamp": datetime.now().isoformat(),
        "samples_analyzed": len(df),
        "features": {}
    }
    
    issues = []
    
    for col in feature_cols:
        # Missing values
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df)
        
        # Outliers (IQR method)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = outliers / len(df)
        
        # Range validation (medical plausibility)
        range_issues = 0
        if col == 'glucose' and ((df[col] < 40) | (df[col] > 300)).any():
            range_issues = ((df[col] < 40) | (df[col] > 300)).sum()
        elif col == 'blood_pressure' and ((df[col] < 30) | (df[col] > 200)).any():
            range_issues = ((df[col] < 30) | (df[col] > 200)).sum()
        elif col == 'bmi' and ((df[col] < 10) | (df[col] > 70)).any():
            range_issues = ((df[col] < 10) | (df[col] > 70)).sum()
        elif col == 'age' and ((df[col] < 18) | (df[col] > 100)).any():
            range_issues = ((df[col] < 18) | (df[col] > 100)).sum()
        
        quality_report["features"][col] = {
            "missing_count": int(missing_count),
            "missing_percentage": round(float(missing_pct), 4),
            "outliers": int(outliers),
            "outlier_percentage": round(float(outlier_pct), 4),
            "range_violations": int(range_issues),
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2)
        }
        
        # Flag issues
        if missing_pct > 0.05:
            issues.append(f"{col}: {missing_pct:.1%} missing values")
        if outlier_pct > 0.15:
            issues.append(f"{col}: {outlier_pct:.1%} outliers")
        if range_issues > 0:
            issues.append(f"{col}: {range_issues} medically implausible values")
    
    quality_report["issues"] = issues
    quality_report["quality_score"] = calculate_quality_score(quality_report)
    
    # Save report
    save_data_quality_log(quality_report)
    
    # Generate alerts if needed
    if quality_report["quality_score"] < 0.8:
        log_alert("DATA_QUALITY", {
            "quality_score": quality_report["quality_score"],
            "issues": issues
        })
    
    return quality_report


def calculate_quality_score(report):
    """Calculate overall data quality score (0-1)"""
    
    penalties = 0
    max_penalties = len(report["features"]) * 3  # 3 checks per feature
    
    for feature, metrics in report["features"].items():
        if metrics["missing_percentage"] > 0.05:
            penalties += 1
        if metrics["outlier_percentage"] > 0.15:
            penalties += 1
        if metrics["range_violations"] > 0:
            penalties += 1
    
    score = 1 - (penalties / max_penalties)
    return round(float(score), 4)


def save_data_quality_log(report):
    """Append data quality report to log"""
    
    logs = []
    if os.path.exists(DATA_QUALITY_LOG_PATH):
        with open(DATA_QUALITY_LOG_PATH) as f:
            logs = json.load(f)
    
    logs.append(report)
    
    # Keep last 30 reports
    logs = logs[-30:]
    
    with open(DATA_QUALITY_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


# ══════════════════════════════════════════════════════════
# SYSTEM HEALTH MONITORING
# ══════════════════════════════════════════════════════════

def monitor_system_health():
    """
    Monitor system operational metrics:
    - Prediction volume trends
    - Response time patterns
    - Error rates
    - System uptime
    """
    
    if not os.path.exists(DB_PATH):
        return {"error": "No audit log found"}
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT id, timestamp, prediction, confidence
        FROM audit_log
        ORDER BY timestamp DESC
    """, conn)
    conn.close()
    
    if len(df) == 0:
        return {"error": "No predictions to monitor"}
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    now = datetime.now()
    
    # Volume metrics
    last_hour = len(df[df['timestamp'] > now - timedelta(hours=1)])
    last_24h = len(df[df['timestamp'] > now - timedelta(hours=24)])
    last_7d = len(df[df['timestamp'] > now - timedelta(days=7)])
    
    # Time-based patterns
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour').size().mean()
    
    # Calculate prediction rate (predictions per hour)
    if len(df) > 1:
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        avg_rate = len(df) / time_span if time_span > 0 else 0
    else:
        avg_rate = 0
    
    # Confidence trends (proxy for model certainty)
    avg_confidence = df['confidence'].mean()
    recent_confidence = df.head(50)['confidence'].mean()
    confidence_trend = "STABLE"
    if recent_confidence < avg_confidence - 0.05:
        confidence_trend = "DECREASING"
    elif recent_confidence > avg_confidence + 0.05:
        confidence_trend = "INCREASING"
    
    health = {
        "timestamp": now.isoformat(),
        "volume_metrics": {
            "last_hour": int(last_hour),
            "last_24h": int(last_24h),
            "last_7days": int(last_7d),
            "total_predictions": int(len(df)),
            "avg_per_hour": round(float(hourly_avg), 2),
            "prediction_rate": round(float(avg_rate), 2)
        },
        "confidence_metrics": {
            "overall_avg": round(float(avg_confidence), 4),
            "recent_avg": round(float(recent_confidence), 4),
            "trend": confidence_trend
        },
        "operational_status": "HEALTHY" if last_24h > 0 else "INACTIVE",
        "uptime_days": (now - df['timestamp'].min()).days
    }
    
    return health


# ══════════════════════════════════════════════════════════
# AUTOMATED RETRAINING TRIGGERS
# ══════════════════════════════════════════════════════════

def evaluate_retraining_need():
    """
    Decide if model should be retrained based on:
    1. Performance degradation
    2. Data drift
    3. Data volume (enough new data)
    4. Time since last training
    """
    
    triggers = []
    should_retrain = False
    
    # Check 1: Performance degradation
    perf_check = detect_performance_degradation()
    if perf_check.get("degradation_detected", False):
        triggers.append({
            "type": "PERFORMANCE_DEGRADATION",
            "severity": perf_check["severity"],
            "details": f"Accuracy dropped from {perf_check['baseline_accuracy']:.2%} to {perf_check['current_accuracy']:.2%}"
        })
        should_retrain = True
    
    # Check 2: Data drift (load from drift monitor)
    drift_alerts_path = "logs/drift_alerts.json"
    if os.path.exists(drift_alerts_path):
        with open(drift_alerts_path) as f:
            content = f.read().strip()
            if content:
                drift_alerts = json.loads(content)
                if drift_alerts and drift_alerts[-1]["severity"] in ["HIGH", "CRITICAL"]:
                    triggers.append({
                        "type": "DATA_DRIFT",
                        "severity": drift_alerts[-1]["severity"],
                        "details": f"Drift detected in {len(drift_alerts[-1]['drifted_features'])} features"
                    })
                    should_retrain = True
    
    # Check 3: Data volume
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        total = pd.read_sql("SELECT COUNT(*) as cnt FROM audit_log", conn).iloc[0]['cnt']
        conn.close()
        
        if total > 500:  # Enough data for retraining
            triggers.append({
                "type": "SUFFICIENT_DATA",
                "severity": "INFO",
                "details": f"{total} predictions logged (threshold: 500)"
            })
    
    # Check 4: Data quality issues
    quality = monitor_data_quality()
    if not isinstance(quality, dict) or "error" not in quality:
        if quality.get("quality_score", 1.0) < 0.7:
            triggers.append({
                "type": "DATA_QUALITY_ISSUES",
                "severity": "HIGH",
                "details": f"Quality score: {quality['quality_score']:.2%}"
            })
            should_retrain = True
    
    recommendation = {
        "timestamp": datetime.now().isoformat(),
        "should_retrain": should_retrain,
        "triggers": triggers,
        "recommendation": generate_retrain_recommendation(triggers, should_retrain)
    }
    
    if should_retrain:
        log_alert("RETRAIN_RECOMMENDED", recommendation)
    
    return recommendation


def generate_retrain_recommendation(triggers, should_retrain):
    """Generate human-readable retraining recommendation"""
    
    if not should_retrain:
        return "No retraining needed. Model performance is stable."
    
    critical = [t for t in triggers if t["severity"] in ["CRITICAL", "HIGH"]]
    
    if len(critical) >= 2:
        return f"URGENT: Multiple critical issues detected ({len(critical)}). Immediate retraining recommended."
    elif len(critical) == 1:
        return f"Retraining recommended due to: {critical[0]['details']}"
    else:
        return "Retraining suggested to refresh model with recent data patterns."


# ══════════════════════════════════════════════════════════
# ALERT LOGGING
# ══════════════════════════════════════════════════════════

def log_alert(alert_type, details):
    """Log production alerts for dashboard display"""
    
    alerts = []
    if os.path.exists(ALERTS_LOG_PATH):
        with open(ALERTS_LOG_PATH) as f:
            content = f.read().strip()
            if content:
                alerts = json.loads(content)
    
    alert = {
        "timestamp": datetime.now().isoformat(),
        "type": alert_type,
        "details": details
    }
    
    alerts.append(alert)
    
    # Keep last 50 alerts
    alerts = alerts[-50:]
    
    with open(ALERTS_LOG_PATH, "w") as f:
        json.dump(alerts, f, indent=2)


def load_alerts():
    """Load recent production alerts"""
    if not os.path.exists(ALERTS_LOG_PATH):
        return []
    
    with open(ALERTS_LOG_PATH) as f:
        content = f.read().strip()
        return json.loads(content) if content else []


# ══════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  PRODUCTION MONITORING REPORT")
    print("="*70)
    
    # Performance
    print("\n📊 PERFORMANCE TRACKING:")
    print("-" * 70)
    perf = calculate_rolling_performance()
    if "error" not in perf:
        curr = perf["current_metrics"]
        print(f"Current Accuracy:     {curr['accuracy']:.2%}")
        print(f"Avg Confidence:       {curr['avg_confidence']:.2%}")
        print(f"Predictions (24h):    {curr['predictions_last_24h']}")
        print(f"Total Predictions:    {curr['total_predictions']}")
    
    # Data Quality
    print("\n🔍 DATA QUALITY:")
    print("-" * 70)
    quality = monitor_data_quality()
    if "error" not in quality:
        print(f"Quality Score:        {quality['quality_score']:.1%}")
        if quality['issues']:
            print("\nIssues Detected:")
            for issue in quality['issues']:
                print(f"  ⚠️  {issue}")
        else:
            print("✅ No quality issues detected")
    
    # System Health
    print("\n💚 SYSTEM HEALTH:")
    print("-" * 70)
    health = monitor_system_health()
    if "error" not in health:
        vol = health["volume_metrics"]
        print(f"Status:               {health['operational_status']}")
        print(f"Uptime:               {health['uptime_days']} days")
        print(f"Prediction Rate:      {vol['prediction_rate']:.1f}/hour")
        print(f"Last 24h:             {vol['last_24h']} predictions")
    
    # Retraining
    print("\n🔄 RETRAINING ASSESSMENT:")
    print("-" * 70)
    retrain = evaluate_retraining_need()
    print(f"Should Retrain:       {'YES ⚠️' if retrain['should_retrain'] else 'NO ✅'}")
    print(f"Recommendation:       {retrain['recommendation']}")
    if retrain['triggers']:
        print("\nTriggers:")
        for trigger in retrain['triggers']:
            print(f"  [{trigger['severity']}] {trigger['type']}: {trigger['details']}")
    
    print("\n" + "="*70)