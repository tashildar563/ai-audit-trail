from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/ai_governance")

engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Connection pool for concurrency
    max_overflow=40,        # Allow up to 60 total connections
    pool_pre_ping=True,     # Verify connections before use
    pool_recycle=3600       # Recycle connections every hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ══════════════════════════════════════════════════════════
# DATABASE MODELS (ORM)
# ══════════════════════════════════════════════════════════

class Client(Base):
    __tablename__ = "clients"
    
    client_id = Column(String, primary_key=True, index=True)
    company_name = Column(String, nullable=False, unique=True)  # Add unique=True
    email = Column(String, unique=True, index=True)  # Add this line
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
    
    # Relationships
    predictions = relationship("Prediction", back_populates="client", cascade="all, delete-orphan")
    fairness_audits = relationship("FairnessAudit", back_populates="client")
    drift_alerts = relationship("DriftAlert", back_populates="client")


class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), index=True, nullable=False)
    prediction_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String, index=True)
    features = Column(JSON)
    prediction = Column(JSON)
    confidence = Column(Float)
    prediction_metadata = Column("metadata", JSON)
    
    # Partitioning hint (for future sharding)
    date_partition = Column(String, index=True)  # YYYY-MM-DD for daily partitions
    
    # Relationships
    client = relationship("Client", back_populates="predictions")


class FairnessAudit(Base):
    __tablename__ = "fairness_audits"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String, index=True)
    protected_attribute = Column(String)
    metrics = Column(JSON)
    status = Column(String)  # FAIR, BIAS_DETECTED
    
    client = relationship("Client", back_populates="fairness_audits")


class DriftAlert(Base):
    __tablename__ = "drift_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String, index=True)
    feature = Column(String)
    drift_score = Column(Float)
    severity = Column(String)  # LOW, MEDIUM, HIGH, CRITICAL
    
    client = relationship("Client", back_populates="drift_alerts")


# ══════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ══════════════════════════════════════════════════════════

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Add to existing imports
from sqlalchemy import Float, Text

# ... existing code ...

# Add after existing models (Client, Prediction, etc.)

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


# Update Client model to add relationships
# Add these lines to the Client class:
performance_metrics = relationship("PerformanceMetrics", back_populates="client", cascade="all, delete-orphan")
performance_alerts = relationship("PerformanceAlert", back_populates="client", cascade="all, delete-orphan")


if __name__ == "__main__":
    init_db()