from src.celery_app import celery_app
from src.database import SessionLocal, Prediction, FairnessAudit, DriftAlert, Client
from scipy import stats
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta


@celery_app.task(name="tasks.run_fairness_analysis")
def run_fairness_analysis(client_id: str, model_name: str, protected_attr: str, lookback_days: int = 7):
    """
    Background task to run fairness analysis
    Heavy computation moved off the request thread
    """
    db = SessionLocal()
    
    try:
        # Fetch predictions
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        predictions = db.query(Prediction).filter(
            Prediction.client_id == client_id,
            Prediction.model_name == model_name,
            Prediction.timestamp >= cutoff_date
        ).all()
        
        if len(predictions) < 10:
            return {"error": "Insufficient data"}
        
        # Convert to DataFrame
        data = []
        for pred in predictions:
            features = pred.features
            prediction = pred.prediction
            
            data.append({
                "protected_attr": features.get(protected_attr),
                "prediction_class": prediction.get("class"),
                "confidence": pred.confidence
            })
        
        df = pd.DataFrame(data)
        df = df.dropna(subset=["protected_attr"])
        
        # Calculate metrics by group
        groups = df.groupby("protected_attr")
        metrics_by_group = {}
        
        for group_name, group_df in groups:
            positive_rate = (group_df["prediction_class"] == "approved").mean()
            
            metrics_by_group[str(group_name)] = {
                "sample_size": len(group_df),
                "positive_rate": float(positive_rate),
                "avg_confidence": float(group_df["confidence"].mean())
            }
        
        # Disparate impact
        rates = [m["positive_rate"] for m in metrics_by_group.values()]
        disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0
        passes_80_rule = 0.8 <= disparate_impact <= 1.25
        
        # Save audit
        audit = FairnessAudit(
            client_id=client_id,
            model_name=model_name,
            protected_attribute=protected_attr,
            metrics={
                "groups": metrics_by_group,
                "disparate_impact": disparate_impact,
                "passes_80_rule": passes_80_rule
            },
            status="FAIR" if passes_80_rule else "BIAS_DETECTED"
        )
        db.add(audit)
        db.commit()
        
        return {
            "status": "completed",
            "audit_id": audit.id,
            "result": "FAIR" if passes_80_rule else "BIAS_DETECTED"
        }
    
    finally:
        db.close()


@celery_app.task(name="tasks.detect_feature_drift")
def detect_feature_drift(client_id: str, model_name: str, feature: str, 
                        lookback_days: int = 7, reference_days: int = 30):
    """
    Background task for drift detection
    """
    db = SessionLocal()
    
    try:
        # Fetch recent data
        recent_cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent_predictions = db.query(Prediction).filter(
            Prediction.client_id == client_id,
            Prediction.model_name == model_name,
            Prediction.timestamp >= recent_cutoff
        ).all()
        
        # Fetch reference data
        reference_start = datetime.utcnow() - timedelta(days=reference_days + lookback_days)
        reference_end = datetime.utcnow() - timedelta(days=lookback_days)
        reference_predictions = db.query(Prediction).filter(
            Prediction.client_id == client_id,
            Prediction.model_name == model_name,
            Prediction.timestamp >= reference_start,
            Prediction.timestamp < reference_end
        ).all()
        
        if len(recent_predictions) < 5 or len(reference_predictions) < 5:
            return {"error": "Insufficient data"}
        
        # Extract feature values
        recent_values = [p.features.get(feature) for p in recent_predictions]
        reference_values = [p.features.get(feature) for p in reference_predictions]
        
        # Remove None
        recent_values = [v for v in recent_values if v is not None]
        reference_values = [v for v in reference_values if v is not None]
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(reference_values, recent_values)
        
        drift_detected = p_value < 0.05
        severity = "HIGH" if p_value < 0.01 else "MEDIUM" if drift_detected else "LOW"
        
        if drift_detected:
            alert = DriftAlert(
                client_id=client_id,
                model_name=model_name,
                feature=feature,
                drift_score=float(ks_stat),
                severity=severity
            )
            db.add(alert)
            db.commit()
        
        return {
            "status": "completed",
            "drift_detected": drift_detected,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "severity": severity
        }
    
    finally:
        db.close()


@celery_app.task(name="tasks.sync_usage_to_db")
def sync_usage_to_db():
    """
    Periodic task to sync Redis usage counters to PostgreSQL
    Run every 5 minutes via Celery Beat
    """
    from src.cache import redis_client
    
    db = SessionLocal()
    
    try:
        # Get all usage keys from Redis
        for key in redis_client.scan_iter("usage:*"):
            client_id = key.split(":")[1]
            count = int(redis_client.get(key))
            
            # Update database
            client = db.query(Client).filter(Client.client_id == client_id).first()
            if client:
                client.usage_count = count
        
        db.commit()
        return {"status": "synced"}
    
    finally:
        db.close()


@celery_app.task(name="tasks.cleanup_old_data")
def cleanup_old_data(retention_days: int = 90):
    """
    Clean up predictions older than retention period
    Run daily via Celery Beat
    """
    db = SessionLocal()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        deleted = db.query(Prediction).filter(
            Prediction.timestamp < cutoff_date
        ).delete()
        
        db.commit()
        
        return {"status": "cleaned", "deleted_predictions": deleted}
    
    finally:
        db.close()