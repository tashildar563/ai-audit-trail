from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import secrets
from datetime import datetime, timedelta
import os

from src.database import get_db, Client, Prediction, FairnessAudit, init_db

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


@app.post("/api/v1/login")
async def login_client(request: LoginRequest, db: Session = Depends(get_db)):
    """Login and retrieve API key"""
    
    # Find client by email and company name
    client = db.query(Client).filter(
        Client.email == request.email,
        Client.company_name == request.company_name
    ).first()
    
    if not client:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials. Please check your email and company name."
        )
    
    if not client.is_active:
        raise HTTPException(
            status_code=403,
            detail="Account is inactive. Please contact support."
        )
    
    return {
        "client_id": client.client_id,
        "api_key": client.api_key,
        "company_name": client.company_name,
        "plan": client.plan,
        "message": "Login successful!"
    }


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
    """Register new client with duplicate prevention"""
    
    # Check if company name already exists
    existing_company = db.query(Client).filter(
        Client.company_name == request.company_name
    ).first()
    
    if existing_company:
        raise HTTPException(
            status_code=400, 
            detail=f"Company name '{request.company_name}' is already registered. Please use a different name or contact support to recover your API key."
        )
    
    # Check if email already exists
    existing_email = db.query(Client).filter(
        Client.email == request.email
    ).first()
    
    if existing_email:
        raise HTTPException(
            status_code=400,
            detail=f"Email '{request.email}' is already registered. Please use a different email or login to retrieve your API key."
        )
    
    client_id = f"client_{secrets.token_urlsafe(8)}"
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    
    usage_limits = {"free": 1000, "pro": 10000, "enterprise": 100000}
    
    client = Client(
        client_id=client_id,
        company_name=request.company_name,
        api_key=api_key,
        api_key_hash=secrets.token_hex(32),
        plan=request.plan,
        usage_limit=usage_limits.get(request.plan, 1000),
        email=request.email  # Make sure to add email field to Client model
    )
    
    db.add(client)
    db.commit()
    db.refresh(client)
    
    return {
        "client_id": client_id,
        "api_key": api_key,
        "company_name": request.company_name,
        "plan": request.plan,
        "usage_limit": client.usage_limit,
        "message": "Save your API key - it won't be shown again!"
    }


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
    client: Client = Depends(verify_api_key),
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


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    init_db()
    print("✅ Database initialized")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)