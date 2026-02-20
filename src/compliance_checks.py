"""
Compliance Check Functions
Each function checks if a specific requirement is met
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import Prediction, FairnessAudit, DriftAlert, Client


# ══════════════════════════════════════════════════════════
# FAIRNESS CHECKS
# ══════════════════════════════════════════════════════════

def check_has_fairness_monitoring(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if fairness monitoring is in place
    
    Returns:
        {
            "passed": bool,
            "score": int (0-100),
            "evidence": str,
            "details": dict,
            "recommendation": str or None
        }
    """
    # Check for fairness audits in last 90 days
    cutoff = datetime.utcnow() - timedelta(days=90)
    
    recent_audits = db.query(FairnessAudit).filter(
        FairnessAudit.client_id == client_id,
        FairnessAudit.model_name == model_name,
        FairnessAudit.timestamp >= cutoff
    ).all()
    
    if not recent_audits:
        return {
            "passed": False,
            "score": 0,
            "evidence": "No fairness audits found in last 90 days",
            "details": {
                "audit_count": 0,
                "last_audit": None,
                "days_since_audit": None
            },
            "recommendation": "Run fairness analysis immediately. EU AI Act requires bias monitoring at least every 90 days."
        }
    
    # Get most recent audit
    latest_audit = max(recent_audits, key=lambda x: x.timestamp)
    days_since = (datetime.utcnow() - latest_audit.timestamp).days
    
    # Check disparate impact
    disparate_impact = latest_audit.metrics.get('disparate_impact', 0)
    passes_80_rule = disparate_impact >= 0.8
    
    # Calculate score
    score = 100
    if not passes_80_rule:
        score -= 30  # Major penalty for failing 80% rule
    if days_since > 30:
        score -= 20  # Penalty for old audit
    if days_since > 60:
        score -= 20  # Additional penalty
    
    score = max(0, score)
    passed = score >= 70
    
    return {
        "passed": passed,
        "score": score,
        "evidence": f"{len(recent_audits)} fairness audit(s) in last 90 days. Latest: {days_since} days ago. Disparate impact: {disparate_impact:.2f}",
        "details": {
            "audit_count": len(recent_audits),
            "last_audit": latest_audit.timestamp.isoformat(),
            "days_since_audit": days_since,
            "disparate_impact_ratio": round(disparate_impact, 4),
            "passes_80_rule": passes_80_rule,
            "status": latest_audit.status
        },
        "recommendation": None if passed else "Schedule more frequent fairness audits and address bias issues detected."
    }


def check_has_annual_bias_audit(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if annual bias audit exists (NYC Law 144 requirement)
    Must be within last 365 days
    """
    cutoff = datetime.utcnow() - timedelta(days=365)
    
    audits = db.query(FairnessAudit).filter(
        FairnessAudit.client_id == client_id,
        FairnessAudit.model_name == model_name,
        FairnessAudit.timestamp >= cutoff
    ).all()
    
    if not audits:
        return {
            "passed": False,
            "score": 0,
            "evidence": "No bias audit found in last 365 days",
            "details": {
                "last_audit": None,
                "days_since": None
            },
            "recommendation": "NYC Law 144 requires annual bias audit by independent auditor. Schedule immediately."
        }
    
    latest = max(audits, key=lambda x: x.timestamp)
    days_since = (datetime.utcnow() - latest.timestamp).days
    
    # Score based on recency
    if days_since <= 180:
        score = 100
    elif days_since <= 270:
        score = 80
    elif days_since <= 365:
        score = 60
    else:
        score = 0
    
    return {
        "passed": score >= 60,
        "score": score,
        "evidence": f"Last bias audit: {days_since} days ago",
        "details": {
            "last_audit": latest.timestamp.isoformat(),
            "days_since": days_since,
            "next_due": (latest.timestamp + timedelta(days=365)).isoformat()
        },
        "recommendation": None if score >= 60 else "Bias audit overdue. NYC Law 144 requires annual audits."
    }


def check_calculates_impact_ratios(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if impact ratios are calculated by protected attributes
    NYC Law 144 requires race and gender
    """
    # Check if recent fairness audits exist with required attributes
    cutoff = datetime.utcnow() - timedelta(days=90)
    
    audits = db.query(FairnessAudit).filter(
        FairnessAudit.client_id == client_id,
        FairnessAudit.model_name == model_name,
        FairnessAudit.timestamp >= cutoff
    ).all()
    
    if not audits:
        return {
            "passed": False,
            "score": 0,
            "evidence": "No fairness audits with impact ratios found",
            "details": {},
            "recommendation": "Calculate selection rates by race and gender as required by NYC Law 144."
        }
    
    # Check if audits analyze required protected attributes
    latest = max(audits, key=lambda x: x.timestamp)
    protected_attr = latest.protected_attribute
    
    # For now, we check if ANY protected attribute is analyzed
    # In production, you'd check specifically for race/gender
    has_metrics = latest.metrics and 'groups' in latest.metrics
    
    if has_metrics:
        return {
            "passed": True,
            "score": 100,
            "evidence": f"Impact ratios calculated for '{protected_attr}'",
            "details": {
                "protected_attribute": protected_attr,
                "groups_analyzed": list(latest.metrics.get('groups', {}).keys()),
                "last_calculated": latest.timestamp.isoformat()
            },
            "recommendation": None
        }
    else:
        return {
            "passed": False,
            "score": 30,
            "evidence": f"Incomplete impact ratio analysis",
            "details": {},
            "recommendation": "Ensure selection rates are calculated by race and gender."
        }


# ══════════════════════════════════════════════════════════
# MONITORING & PERFORMANCE CHECKS
# ══════════════════════════════════════════════════════════

def check_has_production_monitoring(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if model is being monitored in production
    (predictions are being tracked)
    """
    # Check for recent predictions
    cutoff = datetime.utcnow() - timedelta(days=30)
    
    prediction_count = db.query(Prediction).filter(
        Prediction.client_id == client_id,
        Prediction.model_name == model_name,
        Prediction.timestamp >= cutoff
    ).count()
    
    if prediction_count == 0:
        return {
            "passed": False,
            "score": 0,
            "evidence": "No predictions tracked in last 30 days",
            "details": {
                "prediction_count": 0,
                "monitoring_active": False
            },
            "recommendation": "Start tracking predictions to enable production monitoring."
        }
    
    # Check for drift monitoring
    drift_checks = db.query(DriftAlert).filter(
        DriftAlert.client_id == client_id,
        DriftAlert.model_name == model_name,
        DriftAlert.timestamp >= cutoff
    ).count()
    
    score = 70  # Base score for having predictions
    if drift_checks > 0:
        score = 100  # Full score if drift monitoring is active
    
    return {
        "passed": True,
        "score": score,
        "evidence": f"{prediction_count} predictions tracked in last 30 days. {drift_checks} drift checks performed.",
        "details": {
            "prediction_count": prediction_count,
            "drift_checks": drift_checks,
            "monitoring_active": True
        },
        "recommendation": None if score == 100 else "Enable drift monitoring for complete production oversight."
    }


def check_tracks_performance(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if model performance is being tracked
    """
    # Check if predictions exist (basic requirement)
    total_predictions = db.query(Prediction).filter(
        Prediction.client_id == client_id,
        Prediction.model_name == model_name
    ).count()
    
    if total_predictions == 0:
        return {
            "passed": False,
            "score": 0,
            "evidence": "No predictions tracked",
            "details": {
                "total_predictions": 0,
                "performance_tracking": False
            },
            "recommendation": "Start tracking predictions to monitor model performance."
        }
    
    # In a real system, you'd check for actual outcomes vs predictions
    # For now, we just check if predictions are being logged
    
    return {
        "passed": True,
        "score": 80,  # Partial score (would be 100 with actual performance metrics)
        "evidence": f"{total_predictions} predictions tracked. Performance monitoring active.",
        "details": {
            "total_predictions": total_predictions,
            "performance_tracking": True
        },
        "recommendation": "Consider tracking actual outcomes to calculate accuracy, precision, and recall."
    }


# ══════════════════════════════════════════════════════════
# DOCUMENTATION & GOVERNANCE CHECKS
# ══════════════════════════════════════════════════════════

def check_has_documentation(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if model has basic documentation
    (In production, this would check for model cards, data documentation, etc.)
    """
    # For MVP, we check if model exists and has been used
    predictions = db.query(Prediction).filter(
        Prediction.client_id == client_id,
        Prediction.model_name == model_name
    ).count()
    
    if predictions > 0:
        return {
            "passed": True,
            "score": 60,  # Partial score (would need actual model card for 100)
            "evidence": f"Model '{model_name}' is registered and actively used",
            "details": {
                "has_model_card": False,  # Would check actual model card in production
                "predictions_count": predictions
            },
            "recommendation": "Create comprehensive model card documenting training data, architecture, and performance metrics."
        }
    else:
        return {
            "passed": False,
            "score": 0,
            "evidence": "Model not found or not in use",
            "details": {},
            "recommendation": "Register model and document its specifications."
        }


def check_has_human_oversight(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if human oversight process exists
    (In production, this would check for actual review processes)
    """
    # For MVP, we assume if they're using the platform, they have some oversight
    # In production, you'd verify actual review workflows
    
    return {
        "passed": False,  # Default to not passed for now
        "score": 0,
        "evidence": "Human oversight process not documented in system",
        "details": {
            "has_review_process": False
        },
        "recommendation": "Document and implement human review process for high-stakes predictions. EU AI Act requires human oversight for high-risk systems."
    }


def check_has_transparency_notice(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if users are notified about AI usage
    """
    # For MVP, this can't be automatically verified
    # In production, you'd verify actual notification mechanisms
    
    return {
        "passed": False,
        "score": 0,
        "evidence": "Transparency notice not verified in system",
        "details": {
            "has_notification": False
        },
        "recommendation": "Implement and document user notification that AI is being used in decisions."
    }


def check_public_disclosure(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if audit results are publicly disclosed (NYC Law 144)
    """
    # For MVP, this can't be automatically verified
    
    return {
        "passed": False,
        "score": 0,
        "evidence": "Public disclosure not verified",
        "details": {},
        "recommendation": "Publish bias audit results on company website as required by NYC Law 144."
    }


def check_candidate_notification(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if candidates are notified (NYC Law 144)
    """
    return {
        "passed": False,
        "score": 0,
        "evidence": "Candidate notification not verified",
        "details": {},
        "recommendation": "Implement notification system to inform candidates that AI is used in hiring decisions."
    }


def check_alternative_process(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if alternative selection process is available (NYC Law 144)
    """
    return {
        "passed": False,
        "score": 0,
        "evidence": "Alternative process not documented",
        "details": {},
        "recommendation": "Provide alternative selection process option for candidates who request it."
    }


def check_data_governance(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check data governance practices
    """
    return {
        "passed": False,
        "score": 0,
        "evidence": "Data governance not documented",
        "details": {},
        "recommendation": "Document training data sources, quality checks, and representativeness."
    }


def check_risk_assessment(db: Session, client_id: str, model_name: str) -> dict:
    """
    Check if risk assessment has been performed
    """
    return {
        "passed": False,
        "score": 0,
        "evidence": "Risk assessment not documented",
        "details": {},
        "recommendation": "Conduct and document risk assessment for AI system as required by EU AI Act."
    }


# ══════════════════════════════════════════════════════════
# HELPER: Get check function by name
# ══════════════════════════════════════════════════════════

def get_check_function(function_name: str):
    """
    Get a check function by its name
    
    Args:
        function_name: Name of the check function
    
    Returns:
        Function object or None
    """
    # Map function names to actual functions
    FUNCTION_MAP = {
        "check_has_fairness_monitoring": check_has_fairness_monitoring,
        "check_has_annual_bias_audit": check_has_annual_bias_audit,
        "check_calculates_impact_ratios": check_calculates_impact_ratios,
        "check_has_production_monitoring": check_has_production_monitoring,
        "check_tracks_performance": check_tracks_performance,
        "check_has_documentation": check_has_documentation,
        "check_has_human_oversight": check_has_human_oversight,
        "check_has_transparency_notice": check_has_transparency_notice,
        "check_public_disclosure": check_public_disclosure,
        "check_candidate_notification": check_candidate_notification,
        "check_alternative_process": check_alternative_process,
        "check_data_governance": check_data_governance,
        "check_risk_assessment": check_risk_assessment,
    }
    
    return FUNCTION_MAP.get(function_name)