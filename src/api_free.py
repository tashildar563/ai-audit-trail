from fastapi import FastAPI, Depends, HTTPException, Header, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import secrets
from datetime import datetime, timedelta
import os
from scipy import stats
import numpy as np
import pandas as pd

from src.database import get_db, Client, Prediction, FairnessAudit, init_db
from src.compliance_engine import ComplianceEngine, generate_compliance_summary
from src.regulations_db import get_all_regulations, get_regulations_by_use_case
# Add these imports at the top
from src.auth import (
    hash_password, 
    verify_password, 
    create_new_session, 
    verify_session,
    invalidate_existing_session
)

# Initialize FastAPI
app = FastAPI(
    title="AI Governance API - Free Tier",
    description="Production ML monitoring platform",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    company_name: str
    email: Optional[str] = None
    password: str
    plan: str = "free"


class TrackPredictionRequest(BaseModel):
    prediction_id: Optional[str] = None
    model_name: str
    features: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}


# ══════════════════════════════════════════════════════════
# SIMPLE AUTH (No Redis for free tier)
# ══════════════════════════════════════════════════════════

def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db)
) -> Client:
    """Simple API key verification"""
    
    client = db.query(Client).filter(Client.api_key == x_api_key).first()
    
    if not client:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not client.is_active:
        raise HTTPException(status_code=403, detail="Account inactive")
    
    # Simple usage check
    if client.usage_count >= client.usage_limit:
        raise HTTPException(status_code=429, detail="Usage limit exceeded")
    
    return client


# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════class LoginRequest(BaseModel):
    email: str
    company_name: str

class LoginRequest(BaseModel):
 email: str
 company_name: str
 password: str


@app.post("/api/v1/login")
async def login_client(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login with email and password
    
    Request body:
    {
        "email": "dev@acme.com",
        "password": "SecurePassword123!"
    }
    
    Returns:
    - session_token (store this in localStorage)
    - Invalidates any existing session (single session enforcement)
    """
    try:
        # Find client by email
        client = db.query(Client).filter(
            Client.email == request.email
        ).first()
        
        if not client:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not client.password_hash or not verify_password(request.password, client.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Check if account is active
        if not client.is_active:
            raise HTTPException(
                status_code=403,
                detail="Account is inactive. Please contact support."
            )
        
        # Create new session (invalidates any existing session)
        session_data = create_new_session(db, client)
        
        return {
            "status": "success",
            "session_token": session_data['session_token'],
            "expires_at": session_data['expires_at'],
            "client_id": client.client_id,
            "email": client.email,
            "company_name": client.company_name,
            "plan": client.plan,
            "message": "Login successful"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/logout")
async def logout_client(
    session_token: str = Header(..., alias="X-Session-Token"),
    db: Session = Depends(get_db)
):
    """
    Logout (invalidate session)
    
    Headers:
    X-Session-Token: your_session_token
    """
    try:
        client = verify_session(db, session_token)
        
        if client:
            invalidate_existing_session(db, client.client_id)
            return {"status": "success", "message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=401, detail="Invalid session")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/v1/session/verify")
async def verify_session_endpoint(
    session_token: str = Header(..., alias="X-Session-Token"),
    db: Session = Depends(get_db)
):
    """
    Verify if session is still valid
    
    Headers:
    X-Session-Token: your_session_token
    
    Returns client info if valid, 401 if invalid
    """
    try:
        client = verify_session(db, session_token)
        
        if not client:
            raise HTTPException(status_code=401, detail="Session expired or invalid")
        
        return {
            "valid": True,
            "client_id": client.client_id,
            "email": client.email,
            "company_name": client.company_name,
            "plan": client.plan,
            "session_expires_at": client.session_expires_at.isoformat() if client.session_expires_at else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Update the dependency for dashboard endpoints
def verify_session_token(
    session_token: str = Header(..., alias="X-Session-Token"),
    db: Session = Depends(get_db)
) -> Client:
    """
    Dependency to verify session token for dashboard endpoints
    """
    client = verify_session(db, session_token)
    
    if not client:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please login again."
        )
    
    return client
   
@app.get("/")
async def root():
    """API info"""
    return {
        "name": "AI Governance API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "POST /api/v1/register": "Create account",
            "POST /api/v1/track_prediction": "Log prediction",
            "GET /api/v1/usage": "Check usage",
            "GET /api/v1/dashboard": "Get dashboard data"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/api/v1/register")
async def register_client(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register new client with password
    
    Request body:
    {
        "company_name": "Acme AI Solutions",
        "email": "dev@acme.com",
        "password": "SecurePassword123!",
        "plan": "free"
    }
    
    Returns:
    - client_id
    - api_key (for API access)
    - session_token (for dashboard access)
    """
    try:
        # Validate password strength
        if len(request.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long"
            )
        
        # Check if company name already exists
        existing_company = db.query(Client).filter(
            Client.company_name == request.company_name
        ).first()
        
        if existing_company:
            raise HTTPException(
                status_code=400,
                detail=f"Company name '{request.company_name}' is already registered"
            )
        
        # Check if email already exists
        existing_email = db.query(Client).filter(
            Client.email == request.email
        ).first()
        
        if existing_email:
            raise HTTPException(
                status_code=400,
                detail=f"Email '{request.email}' is already registered"
            )
        
        # Generate IDs
        client_id = f"client_{secrets.token_urlsafe(8)}"
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        api_key_hash = secrets.token_hex(32)
        
        # Hash password
        password_hash = hash_password(request.password)
        
        # Set usage limits
        usage_limits = {"free": 1000, "pro": 50000, "enterprise": 100000}
        
        # Create client
        client = Client(
            client_id=client_id,
            company_name=request.company_name,
            email=request.email,
            api_key=api_key,
            api_key_hash=api_key_hash,
            password_hash=password_hash,
            plan=request.plan,
            usage_limit=usage_limits.get(request.plan, 1000)
        )
        
        db.add(client)
        db.commit()
        db.refresh(client)
        
        # Create initial session
        session_data = create_new_session(db, client)
        
        return {
            "client_id": client_id,
            "api_key": api_key,
            "session_token": session_data['session_token'],
            "expires_at": session_data['expires_at'],
            "company_name": request.company_name,
            "email": request.email,
            "plan": request.plan,
            "usage_limit": client.usage_limit,
            "message": "Registration successful! Save your API key - it won't be shown again."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/track_prediction")
async def track_prediction(
    request: TrackPredictionRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Track a prediction"""
    
    prediction_id = request.prediction_id or f"pred_{secrets.token_urlsafe(8)}"
    
    prediction = Prediction(
        client_id=client.client_id,
        prediction_id=prediction_id,
        model_name=request.model_name,
        features=request.features,
        prediction=request.prediction,
        confidence=request.confidence or request.prediction.get("probability"),
        metadata=request.metadata,
        date_partition=datetime.utcnow().strftime("%Y-%m-%d")
    )
    
    db.add(prediction)
    
    # Update usage count
    client.usage_count += 1
    
    db.commit()
    
    return {
        "status": "tracked",
        "prediction_id": prediction_id,
        "timestamp": prediction.timestamp.isoformat()
    }


@app.get("/api/v1/usage")
async def get_usage(client: Client = Depends(verify_api_key)):
    """Get usage stats"""
    
    return {
        "client_id": client.client_id,
        "company_name": client.company_name,
        "plan": client.plan,
        "usage": {
            "current": client.usage_count,
            "limit": client.usage_limit,
            "remaining": max(0, client.usage_limit - client.usage_count),
            "percentage": round(client.usage_count / client.usage_limit * 100, 2)
        }
    }


@app.get("/api/v1/dashboard")
async def get_dashboard(
    client: Client = Depends(verify_session_token),
    db: Session = Depends(get_db)
):
    """Get dashboard summary"""
    
    # Total predictions
    total = db.query(Prediction).filter(Prediction.client_id == client.client_id).count()
    
    # Recent predictions (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent = db.query(Prediction).filter(
        Prediction.client_id == client.client_id,
        Prediction.timestamp >= week_ago
    ).count()
    
    # Models
    models = db.query(Prediction.model_name).filter(
        Prediction.client_id == client.client_id
    ).distinct().all()
    
    return {
        "client_id": client.client_id,
        "company_name": client.company_name,
        "summary": {
            "total_predictions": total,
            "predictions_last_7_days": recent,
            "models_tracked": len(models),
            "usage_percentage": round(client.usage_count / client.usage_limit * 100, 1)
        },
        "models": [m[0] for m in models]
    }

# ... existing imports and code ...

class FairnessCheckRequest(BaseModel):
    model_name: str
    protected_attribute: str
    lookback_days: int = 7


class DriftCheckRequest(BaseModel):
    model_name: str
    feature: str
    lookback_days: int = 7
    reference_days: int = 30


@app.post("/api/v1/check_fairness")
async def check_fairness(
    request: FairnessCheckRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Run fairness analysis on recent predictions
    """
    from datetime import timedelta
    
    # Fetch recent predictions
    cutoff_date = datetime.utcnow() - timedelta(days=request.lookback_days)
    
    predictions = db.query(Prediction).filter(
        Prediction.client_id == client.client_id,
        Prediction.model_name == request.model_name,
        Prediction.timestamp >= cutoff_date
    ).all()
    
    if len(predictions) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 predictions for fairness analysis. Found: {len(predictions)}"
        )
    
    # Extract data
    data = []
    for pred in predictions:
        protected_value = pred.features.get(request.protected_attribute)
        prediction_class = pred.prediction.get('class')
        
        if protected_value is not None and prediction_class is not None:
            data.append({
                'protected_attr': str(protected_value),
                'prediction': prediction_class,
                'confidence': pred.confidence
            })
    
    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"Protected attribute '{request.protected_attribute}' not found in predictions"
        )
    
    df = pd.DataFrame(data)
    
    # Calculate metrics by group
    groups = df.groupby('protected_attr')
    metrics_by_group = {}
    
    for group_name, group_df in groups:
        total = len(group_df)
        
        # Assume positive class is any prediction that's not "denied" or "no" or similar
        positive_predictions = group_df[
            ~group_df['prediction'].str.lower().isin(['denied', 'no', 'rejected', 'negative', 'low risk'])
        ]
        positive_rate = len(positive_predictions) / total if total > 0 else 0
        
        metrics_by_group[group_name] = {
            'sample_size': total,
            'positive_rate': round(float(positive_rate), 4),
            'avg_confidence': round(float(group_df['confidence'].mean()), 4)
        }
    
    # Calculate disparate impact
    rates = [m['positive_rate'] for m in metrics_by_group.values() if m['positive_rate'] > 0]
    if len(rates) < 2:
        disparate_impact = 1.0
    else:
        disparate_impact = min(rates) / max(rates)
    
    passes_80_rule = 0.8 <= disparate_impact <= 1.25
    
    # Save to database
    fairness_audit = FairnessAudit(
        client_id=client.client_id,
        model_name=request.model_name,
        protected_attribute=request.protected_attribute,
        metrics={
            'groups': metrics_by_group,
            'disparate_impact': round(float(disparate_impact), 4),
            'passes_80_rule': passes_80_rule
        },
        status='FAIR' if passes_80_rule else 'BIAS_DETECTED'
    )
    db.add(fairness_audit)
    db.commit()
    
    return {
        'model_name': request.model_name,
        'protected_attribute': request.protected_attribute,
        'analysis_period_days': request.lookback_days,
        'total_predictions': len(predictions),
        'metrics_by_group': metrics_by_group,
        'disparate_impact_ratio': round(float(disparate_impact), 4),
        'passes_80_rule': passes_80_rule,
        'status': 'FAIR' if passes_80_rule else 'BIAS_DETECTED',
        'timestamp': datetime.utcnow().isoformat()
    }


@app.post("/api/v1/detect_drift")
async def detect_drift(
    request: DriftCheckRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Detect data drift in recent predictions
    """
    from datetime import timedelta
    
    # Fetch recent data
    recent_cutoff = datetime.utcnow() - timedelta(days=request.lookback_days)
    recent_predictions = db.query(Prediction).filter(
        Prediction.client_id == client.client_id,
        Prediction.model_name == request.model_name,
        Prediction.timestamp >= recent_cutoff
    ).all()
    
    # Fetch reference data
    reference_start = datetime.utcnow() - timedelta(days=request.reference_days + request.lookback_days)
    reference_end = datetime.utcnow() - timedelta(days=request.lookback_days)
    reference_predictions = db.query(Prediction).filter(
        Prediction.client_id == client.client_id,
        Prediction.model_name == request.model_name,
        Prediction.timestamp >= reference_start,
        Prediction.timestamp < reference_end
    ).all()
    
    if len(recent_predictions) < 5 or len(reference_predictions) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data. Recent: {len(recent_predictions)}, Reference: {len(reference_predictions)}"
        )
    
    # Extract feature values
    recent_values = [p.features.get(request.feature) for p in recent_predictions]
    reference_values = [p.features.get(request.feature) for p in reference_predictions]
    
    # Remove None values
    recent_values = [v for v in recent_values if v is not None]
    reference_values = [v for v in reference_values if v is not None]
    
    if len(recent_values) < 5 or len(reference_values) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Feature '{request.feature}' not found or insufficient data"
        )
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.ks_2samp(reference_values, recent_values)
    
    drift_detected = p_value < 0.05
    severity = 'HIGH' if p_value < 0.01 else 'MEDIUM' if drift_detected else 'LOW'
    
    # Save alert if drift detected
    if drift_detected:
        drift_alert = DriftAlert(
            client_id=client.client_id,
            model_name=request.model_name,
            feature=request.feature,
            drift_score=float(ks_stat),
            severity=severity
        )
        db.add(drift_alert)
        db.commit()
    
    return {
        'model_name': request.model_name,
        'feature': request.feature,
        'drift_detected': drift_detected,
        'ks_statistic': round(float(ks_stat), 4),
        'p_value': round(float(p_value), 4),
        'severity': severity,
        'reference_period_days': request.reference_days,
        'recent_period_days': request.lookback_days,
        'reference_samples': len(reference_values),
        'recent_samples': len(recent_values),
        'timestamp': datetime.utcnow().isoformat()
    }


@app.get("/api/v1/fairness/history")
async def get_fairness_history(
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get fairness audit history"""
    
    audits = db.query(FairnessAudit).filter(
        FairnessAudit.client_id == client.client_id
    ).order_by(FairnessAudit.timestamp.desc()).limit(10).all()
    
    return {
        'audits': [
            {
                'id': audit.id,
                'model_name': audit.model_name,
                'protected_attribute': audit.protected_attribute,
                'status': audit.status,
                'metrics': audit.metrics,
                'timestamp': audit.timestamp.isoformat()
            }
            for audit in audits
        ]
    }


@app.get("/api/v1/drift/history")
async def get_drift_history(
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get drift alert history"""
    
    alerts = db.query(DriftAlert).filter(
        DriftAlert.client_id == client.client_id
    ).order_by(DriftAlert.timestamp.desc()).limit(10).all()
    
    return {
        'alerts': [
            {
                'id': alert.id,
                'model_name': alert.model_name,
                'feature': alert.feature,
                'drift_score': alert.drift_score,
                'severity': alert.severity,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in alerts
        ]
    }

# ... existing code ...

# Add these Pydantic models after existing models

class ComplianceCheckRequest(BaseModel):
    model_name: str
    regulation_ids: list
    use_case: Optional[str] = None


# ══════════════════════════════════════════════════════════
# COMPLIANCE ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/api/v1/compliance/regulations")
async def list_regulations():
    """
    List all available regulations
    
    No authentication required - public information
    """
    try:
        regulations = get_all_regulations()
        
        # Format for API response
        result = []
        for reg_id, regulation in regulations.items():
            result.append({
                "id": reg_id,
                "name": regulation["name"],
                "jurisdiction": regulation["jurisdiction"],
                "status": regulation["status"],
                "effective_date": regulation.get("effective_date"),
                "description": regulation.get("description"),
                "applies_to": regulation.get("applies_to", []),
                "requirements_count": len(regulation.get("requirements", [])),
                "penalties": regulation.get("penalties", {})
            })
        
        return {
            "regulations": result,
            "total_count": len(result)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/compliance/regulations/by-use-case/{use_case}")
async def get_regulations_for_use_case(use_case: str):
    """
    Get regulations applicable to a specific use case
    
    Args:
        use_case: hiring, credit_scoring, healthcare, etc.
    
    No authentication required - public information
    """
    try:
        applicable = get_regulations_by_use_case(use_case)
        
        return {
            "use_case": use_case,
            "applicable_regulations": applicable,
            "count": len(applicable)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compliance/check")
async def run_compliance_check(
    request: ComplianceCheckRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Run comprehensive compliance check
    
    Request body:
    {
        "model_name": "loan_approval_v1",
        "regulation_ids": ["EU_AI_ACT_HIGH_RISK", "NYC_LAW_144"],
        "use_case": "hiring"  (optional)
    }
    
    Returns:
        Complete compliance report with scores, issues, and recommendations
    """
    try:
        # Validate regulation IDs
        all_regs = get_all_regulations()
        invalid_regs = [r for r in request.regulation_ids if r not in all_regs]
        
        if invalid_regs:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid regulation IDs",
                    "invalid_ids": invalid_regs,
                    "available_ids": list(all_regs.keys())
                }
            )
        
        # Run compliance check
        engine = ComplianceEngine(db)
        
        report = engine.check_compliance(
            client_id=client.client_id,
            model_name=request.model_name,
            regulation_ids=request.regulation_ids,
            use_case=request.use_case
        )
        
        # Add summary text
        report["summary_text"] = generate_compliance_summary(report)
        
        # Increment usage
        client.usage_count += 1
        db.commit()
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Compliance check failed: {str(e)}"
        )


@app.get("/api/v1/compliance/history")
async def get_compliance_history(
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """
    Get compliance check history for this client
    
    Note: In this MVP, we return recent fairness audits as proxy
    In production, you'd store compliance reports in a dedicated table
    """
    try:
        # Get recent fairness audits (as a proxy for compliance history)
        audits = db.query(FairnessAudit).filter(
            FairnessAudit.client_id == client.client_id
        ).order_by(FairnessAudit.timestamp.desc()).limit(limit).all()
        
        history = []
        for audit in audits:
            history.append({
                "id": audit.id,
                "model_name": audit.model_name,
                "protected_attribute": audit.protected_attribute,
                "status": audit.status,
                "timestamp": audit.timestamp.isoformat(),
                "metrics": audit.metrics
            })
        
        return {
            "client_id": client.client_id,
            "history": history,
            "count": len(history)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/compliance/report/{model_name}/summary")
async def get_compliance_summary(
    model_name: str,
    regulation_ids: str,  # Comma-separated list
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get a quick compliance summary (lighter than full check)
    
    Example: /api/v1/compliance/report/loan_model/summary?regulation_ids=EU_AI_ACT_HIGH_RISK,NYC_LAW_144
    """
    try:
        # Parse regulation IDs
        reg_list = [r.strip() for r in regulation_ids.split(',')]
        
        # Run quick check
        engine = ComplianceEngine(db)
        report = engine.check_compliance(
            client_id=client.client_id,
            model_name=model_name,
            regulation_ids=reg_list
        )
        
        # Return only summary info
        return {
            "model_name": model_name,
            "overall_score": report["overall_score"],
            "overall_status": report["overall_status"],
            "regulations_checked": report["regulations_checked"],
            "critical_issues_count": report["summary"]["critical_issues_count"],
            "requirements_passed": report["summary"]["requirements_passed"],
            "requirements_total": report["summary"]["total_requirements_checked"],
            "timestamp": report["timestamp"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add these imports at the top
from src.performance_calculator import PerformanceCalculator
from src.database import PerformanceMetrics, PerformanceAlert

# ... existing code ...

# Add these Pydantic models after existing models

class RecordOutcomeRequest(BaseModel):
    prediction_id: str
    actual_outcome: dict
    feedback_notes: Optional[str] = None


class CalculatePerformanceRequest(BaseModel):
    model_name: str
    period_days: int = 30
    thresholds: Optional[dict] = None


# ══════════════════════════════════════════════════════════
# GROUND TRUTH & PERFORMANCE ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/api/v1/track_outcome")
async def record_actual_outcome(
    request: RecordOutcomeRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Record the actual outcome for a prediction (ground truth)
    
    Request body:
    {
        "prediction_id": "pred_abc123",
        "actual_outcome": {
            "class": "denied",  # or "approved", etc.
            "reason": "customer defaulted"  # optional
        },
        "feedback_notes": "Customer defaulted after 3 months"  # optional
    }
    
    Use this endpoint when you know the actual outcome of a prediction.
    For example:
    - Loan prediction: Did customer actually default?
    - Fraud prediction: Was transaction actually fraudulent?
    - Hiring prediction: Did candidate succeed in role?
    """
    try:
        # Find the prediction
        prediction = db.query(Prediction).filter(
            Prediction.client_id == client.client_id,
            Prediction.prediction_id == request.prediction_id
        ).first()
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction '{request.prediction_id}' not found"
            )
        
        # Check if outcome already recorded
        if prediction.actual_outcome is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Outcome already recorded for this prediction",
                    "existing_outcome": prediction.actual_outcome,
                    "recorded_at": prediction.outcome_timestamp.isoformat() if prediction.outcome_timestamp else None
                }
            )
        
        # Update prediction with actual outcome
        prediction.actual_outcome = request.actual_outcome
        prediction.outcome_timestamp = datetime.utcnow()
        prediction.feedback_notes = request.feedback_notes
        
        db.commit()
        db.refresh(prediction)
        
        # Count how many predictions now have outcomes for this model
        total_predictions = db.query(Prediction).filter(
            Prediction.client_id == client.client_id,
            Prediction.model_name == prediction.model_name
        ).count()
        
        predictions_with_outcomes = db.query(Prediction).filter(
            Prediction.client_id == client.client_id,
            Prediction.model_name == prediction.model_name,
            Prediction.actual_outcome.isnot(None)
        ).count()
        
        coverage_percentage = (predictions_with_outcomes / total_predictions * 100) if total_predictions > 0 else 0
        
        return {
            "status": "outcome_recorded",
            "prediction_id": request.prediction_id,
            "model_name": prediction.model_name,
            "predicted": prediction.prediction,
            "actual": request.actual_outcome,
            "timestamp": prediction.outcome_timestamp.isoformat(),
            "coverage": {
                "total_predictions": total_predictions,
                "with_outcomes": predictions_with_outcomes,
                "percentage": round(coverage_percentage, 2)
            },
            "message": f"Outcome recorded. {predictions_with_outcomes}/{total_predictions} predictions now have outcomes for this model."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record outcome: {str(e)}"
        )


@app.post("/api/v1/performance/calculate")
async def calculate_performance_metrics(
    request: CalculatePerformanceRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Calculate performance metrics for a model
    
    Request body:
    {
        "model_name": "loan_approval_v1",
        "period_days": 30,  # optional, default 30
        "thresholds": {  # optional, for alert triggering
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.80
        }
    }
    
    Returns:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion matrix
    - Per-class metrics (for multiclass)
    - Performance alerts (if below thresholds)
    """
    try:
        calculator = PerformanceCalculator(db)
        
        # Calculate metrics
        metrics = calculator.calculate_metrics(
            client_id=client.client_id,
            model_name=request.model_name,
            period_days=request.period_days
        )
        
        # Check for errors
        if 'error' in metrics:
            return {
                "status": "insufficient_data",
                "error": metrics['error'],
                "total_predictions": metrics.get('total_predictions', 0),
                "predictions_with_outcomes": metrics.get('predictions_with_outcomes', 0),
                "message": "Not enough data to calculate performance. Record actual outcomes using /api/v1/track_outcome endpoint."
            }
        
        # Save metrics to database
        calculator.save_metrics(
            client_id=client.client_id,
            model_name=request.model_name,
            metrics=metrics,
            period_days=request.period_days
        )
        
        # Check for performance degradation
        alerts = []
        if request.thresholds:
            alerts = calculator.check_for_degradation(
                client_id=client.client_id,
                model_name=request.model_name,
                current_metrics=metrics,
                thresholds=request.thresholds
            )
        
        # Increment usage
        client.usage_count += 1
        db.commit()
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "period_days": request.period_days,
            "metrics": metrics,
            "alerts": alerts,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Performance calculation failed: {str(e)}"
        )


@app.get("/api/v1/performance/history/{model_name}")
async def get_performance_history(
    model_name: str,
    limit: int = 10,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get historical performance metrics for a model
    
    Returns the last N performance calculations
    """
    try:
        metrics = db.query(PerformanceMetrics).filter(
            PerformanceMetrics.client_id == client.client_id,
            PerformanceMetrics.model_name == model_name
        ).order_by(PerformanceMetrics.calculated_at.desc()).limit(limit).all()
        
        history = []
        for metric in metrics:
            history.append({
                "id": metric.id,
                "period_start": metric.period_start.isoformat(),
                "period_end": metric.period_end.isoformat(),
                "calculated_at": metric.calculated_at.isoformat(),
                "total_predictions": metric.total_predictions,
                "predictions_with_outcomes": metric.predictions_with_outcomes,
                "accuracy": metric.accuracy,
                "precision": metric.precision_score,
                "recall": metric.recall_score,
                "f1_score": metric.f1_score,
                "confusion_matrix": metric.confusion_matrix,
                "class_metrics": metric.class_metrics
            })
        
        return {
            "model_name": model_name,
            "history": history,
            "count": len(history)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.get("/api/v1/performance/alerts")
async def get_performance_alerts(
    status: str = "active",  # active, acknowledged, resolved, all
    limit: int = 20,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get performance alerts
    
    Query params:
    - status: active (default), acknowledged, resolved, all
    - limit: number of alerts to return (default 20)
    """
    try:
        query = db.query(PerformanceAlert).filter(
            PerformanceAlert.client_id == client.client_id
        )
        
        if status != "all":
            query = query.filter(PerformanceAlert.status == status)
        
        alerts = query.order_by(
            PerformanceAlert.triggered_at.desc()
        ).limit(limit).all()
        
        result = []
        for alert in alerts:
            result.append({
                "id": alert.id,
                "model_name": alert.model_name,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "previous_value": alert.previous_value,
                "status": alert.status,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            })
        
        return {
            "alerts": result,
            "count": len(result),
            "status_filter": status
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@app.patch("/api/v1/performance/alerts/{alert_id}")
async def update_alert_status(
    alert_id: int,
    status: str,  # acknowledged, resolved
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Update alert status (acknowledge or resolve)
    
    Path: /api/v1/performance/alerts/123?status=acknowledged
    """
    try:
        alert = db.query(PerformanceAlert).filter(
            PerformanceAlert.id == alert_id,
            PerformanceAlert.client_id == client.client_id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        if status == "acknowledged":
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = client.client_id
        elif status == "resolved":
            alert.status = "resolved"
            alert.resolved_at = datetime.utcnow()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid status. Use 'acknowledged' or 'resolved'"
            )
        
        db.commit()
        
        return {
            "status": "updated",
            "alert_id": alert_id,
            "new_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update alert: {str(e)}"
        )


@app.get("/api/v1/performance/summary")
async def get_performance_summary(
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get performance summary across all models
    
    Returns overview of:
    - Total predictions with outcomes
    - Average accuracy across models
    - Active alerts
    - Models needing attention
    """
    try:
        # Get all models for this client
        models = db.query(Prediction.model_name).filter(
            Prediction.client_id == client.client_id
        ).distinct().all()
        
        model_names = [m[0] for m in models]
        
        summary = {
            "total_models": len(model_names),
            "models": [],
            "overall_stats": {
                "total_predictions": 0,
                "predictions_with_outcomes": 0,
                "coverage_percentage": 0
            },
            "active_alerts": 0
        }
        
        total_preds = 0
        total_with_outcomes = 0
        
        for model_name in model_names:
            # Get latest metrics
            latest_metric = db.query(PerformanceMetrics).filter(
                PerformanceMetrics.client_id == client.client_id,
                PerformanceMetrics.model_name == model_name
            ).order_by(PerformanceMetrics.calculated_at.desc()).first()
            
            # Count predictions
            model_total = db.query(Prediction).filter(
                Prediction.client_id == client.client_id,
                Prediction.model_name == model_name
            ).count()
            
            model_with_outcomes = db.query(Prediction).filter(
                Prediction.client_id == client.client_id,
                Prediction.model_name == model_name,
                Prediction.actual_outcome.isnot(None)
            ).count()
            
            total_preds += model_total
            total_with_outcomes += model_with_outcomes
            
            model_info = {
                "model_name": model_name,
                "total_predictions": model_total,
                "predictions_with_outcomes": model_with_outcomes,
                "coverage_percentage": round(model_with_outcomes / model_total * 100, 2) if model_total > 0 else 0,
                "latest_metrics": None
            }
            
            if latest_metric:
                model_info["latest_metrics"] = {
                    "accuracy": latest_metric.accuracy,
                    "precision": latest_metric.precision_score,
                    "recall": latest_metric.recall_score,
                    "f1_score": latest_metric.f1_score,
                    "calculated_at": latest_metric.calculated_at.isoformat()
                }
            
            summary["models"].append(model_info)
        
        # Overall stats
        summary["overall_stats"]["total_predictions"] = total_preds
        summary["overall_stats"]["predictions_with_outcomes"] = total_with_outcomes
        summary["overall_stats"]["coverage_percentage"] = round(
            total_with_outcomes / total_preds * 100, 2
        ) if total_preds > 0 else 0
        
        # Active alerts count
        active_alerts_count = db.query(PerformanceAlert).filter(
            PerformanceAlert.client_id == client.client_id,
            PerformanceAlert.status == "active"
        ).count()
        
        summary["active_alerts"] = active_alerts_count
        
        return summary
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    init_db()
    print("✅ Database initialized")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)