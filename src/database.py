from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import os

Base = declarative_base()

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ai_governance:dev_password_change_in_prod@localhost:5432/ai_governance"
)


class Client(Base):
    """Client/Company using the platform"""
    __tablename__ = "clients"
    
    client_id = Column(String, primary_key=True, index=True)
    company_name = Column(String, nullable=False, unique=True)
    email = Column(String, unique=True, index=True)
    api_key = Column(String, unique=True, nullable=False, index=True)
    api_key_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    plan = Column(String, default="free")
    is_active = Column(Boolean, default=True)
    usage_limit = Column(Integer, default=1000)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="client", cascade="all, delete-orphan")
    fairness_audits = relationship("FairnessAudit", back_populates="client")
    drift_alerts = relationship("DriftAlert", back_populates="client")
    performance_metrics = relationship("PerformanceMetrics", back_populates="client", cascade="all, delete-orphan")
    performance_alerts = relationship("PerformanceAlert", back_populates="client", cascade="all, delete-orphan")


class Prediction(Base):
    """Store individual predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    prediction_id = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String, nullable=False, index=True)
    features = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)
    confidence = Column(Float)
    metadata = Column(JSON)
    date_partition = Column(String, index=True)
    
    # Ground truth fields
    actual_outcome = Column(JSON)
    outcome_timestamp = Column(DateTime)
    outcome_recorded_by = Column(String)
    feedback_notes = Column(Text)
    
    # Relationships
    client = relationship("Client", back_populates="predictions")


class FairnessAudit(Base):
    """Store fairness audit results"""
    __tablename__ = "fairness_audits"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    model_name = Column(String, nullable=False, index=True)
    protected_attribute = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metrics = Column(JSON)
    status = Column(String)
    
    # Relationships
    client = relationship("Client", back_populates="fairness_audits")


class DriftAlert(Base):
    """Store drift detection alerts"""
    __tablename__ = "drift_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    model_name = Column(String, nullable=False, index=True)
    feature = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    drift_score = Column(Float, nullable=False)
    severity = Column(String)
    
    # Relationships
    client = relationship("Client", back_populates="drift_alerts")


class PerformanceMetrics(Base):
    """Store calculated performance metrics"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    model_name = Column(String, nullable=False, index=True)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Sample sizes
    total_predictions = Column(Integer, nullable=False)
    predictions_with_outcomes = Column(Integer, nullable=False)
    
    # Classification metrics
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    
    # Confusion matrix and per-class metrics
    confusion_matrix = Column(JSON)
    class_metrics = Column(JSON)
    
    # Regression metrics
    mae = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    
    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="performance_metrics")


class PerformanceAlert(Base):
    """Store performance degradation alerts"""
    __tablename__ = "performance_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    model_name = Column(String, nullable=False, index=True)
    
    # Alert details
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    
    # Values
    current_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    previous_value = Column(Float)
    
    # Status
    status = Column(String, default='active')
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String)
    resolved_at = Column(DateTime)
    
    # Timestamps
    triggered_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="performance_alerts")


# Database initialization
def get_engine():
    return create_engine(DATABASE_URL)


def get_db():
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized")