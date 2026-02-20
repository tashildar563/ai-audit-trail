-- Add outcome column to predictions table
ALTER TABLE predictions 
ADD COLUMN actual_outcome JSONB,
ADD COLUMN outcome_timestamp TIMESTAMP,
ADD COLUMN outcome_recorded_by VARCHAR(255),
ADD COLUMN feedback_notes TEXT;

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    
    -- Time period
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    
    -- Sample sizes
    total_predictions INTEGER NOT NULL,
    predictions_with_outcomes INTEGER NOT NULL,
    
    -- Classification metrics (for binary/multiclass)
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    
    -- Confusion matrix (stored as JSON)
    confusion_matrix JSONB,
    
    -- Per-class metrics (for multiclass)
    class_metrics JSONB,
    
    -- Regression metrics (if applicable)
    mae FLOAT,  -- Mean Absolute Error
    rmse FLOAT,  -- Root Mean Squared Error
    r2_score FLOAT,
    
    -- Metadata
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE
);

CREATE INDEX idx_performance_metrics_client_model ON performance_metrics(client_id, model_name);
CREATE INDEX idx_performance_metrics_period ON performance_metrics(period_start, period_end);

-- Create performance alerts table
CREATE TABLE IF NOT EXISTS performance_alerts (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    
    -- Alert details
    alert_type VARCHAR(50) NOT NULL,  -- accuracy_drop, precision_drop, etc.
    severity VARCHAR(20) NOT NULL,  -- critical, high, medium, low
    metric_name VARCHAR(50) NOT NULL,
    
    -- Values
    current_value FLOAT NOT NULL,
    threshold_value FLOAT NOT NULL,
    previous_value FLOAT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',  -- active, acknowledged, resolved
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMP,
    
    -- Timestamps
    triggered_at TIMESTAMP DEFAULT NOW(),
    
    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE
);

CREATE INDEX idx_performance_alerts_client_model ON performance_alerts(client_id, model_name);
CREATE INDEX idx_performance_alerts_status ON performance_alerts(status);

-- Add comment
COMMENT ON TABLE performance_metrics IS 'Stores calculated performance metrics for models over time periods';
COMMENT ON TABLE performance_alerts IS 'Tracks performance degradation alerts';