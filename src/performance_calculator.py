"""
Performance Metrics Calculator
Calculates accuracy, precision, recall, F1, confusion matrix
"""

from sqlalchemy.orm import Session
from src.database import Prediction, PerformanceMetrics, PerformanceAlert
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class PerformanceCalculator:
    """
    Calculate model performance metrics from predictions + outcomes
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_metrics(
        self,
        client_id: str,
        model_name: str,
        period_days: int = 30
    ) -> Dict:
        """
        Calculate performance metrics for a model over a time period
        
        Args:
            client_id: Client ID
            model_name: Model name
            period_days: Number of days to look back
        
        Returns:
            Dictionary with all performance metrics
        """
        
        # Get predictions with actual outcomes
        period_start = datetime.utcnow() - timedelta(days=period_days)
        period_end = datetime.utcnow()
        
        predictions = self.db.query(Prediction).filter(
            Prediction.client_id == client_id,
            Prediction.model_name == model_name,
            Prediction.timestamp >= period_start,
            Prediction.timestamp <= period_end,
            Prediction.actual_outcome.isnot(None)  # Only predictions with outcomes
        ).all()
        
        total_predictions = self.db.query(Prediction).filter(
            Prediction.client_id == client_id,
            Prediction.model_name == model_name,
            Prediction.timestamp >= period_start,
            Prediction.timestamp <= period_end
        ).count()
        
        if len(predictions) == 0:
            return {
                "error": "No predictions with outcomes found",
                "total_predictions": total_predictions,
                "predictions_with_outcomes": 0,
                "coverage": 0
            }
        
        # Extract predicted and actual values
        y_pred = []
        y_true = []
        
        for pred in predictions:
            # Get predicted class
            predicted_class = self._extract_class(pred.prediction)
            # Get actual class
            actual_class = self._extract_class(pred.actual_outcome)
            
            if predicted_class is not None and actual_class is not None:
                y_pred.append(predicted_class)
                y_true.append(actual_class)
        
        if len(y_pred) == 0:
            return {
                "error": "Could not extract classes from predictions/outcomes",
                "total_predictions": total_predictions,
                "predictions_with_outcomes": len(predictions)
            }
        
        # Determine if binary or multiclass
        unique_classes = list(set(y_true + y_pred))
        is_binary = len(unique_classes) == 2
        
        # Calculate metrics
        if is_binary:
            metrics = self._calculate_binary_metrics(y_pred, y_true, unique_classes)
        else:
            metrics = self._calculate_multiclass_metrics(y_pred, y_true, unique_classes)
        
        # Add metadata
        metrics.update({
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_predictions": total_predictions,
            "predictions_with_outcomes": len(predictions),
            "coverage_percentage": round(len(predictions) / total_predictions * 100, 2) if total_predictions > 0 else 0,
            "unique_classes": unique_classes
        })
        
        return metrics
    
    def _extract_class(self, data: dict) -> str:
        """
        Extract class label from prediction/outcome JSON
        Handles various formats: {"class": "X"}, {"label": "X"}, {"prediction": "X"}
        """
        if not data:
            return None
        
        # Try common field names
        for field in ['class', 'label', 'prediction', 'outcome', 'result']:
            if field in data:
                return str(data[field]).lower()
        
        # If dict has only one key, use its value
        if len(data) == 1:
            return str(list(data.values())[0]).lower()
        
        return None
    
    def _calculate_binary_metrics(
        self,
        y_pred: List[str],
        y_true: List[str],
        classes: List[str]
    ) -> Dict:
        """
        Calculate metrics for binary classification
        """
        
        # Ensure we have exactly 2 classes
        if len(classes) != 2:
            classes = classes[:2]
        
        # Determine positive class (usually the second class alphabetically)
        positive_class = sorted(classes)[1]
        negative_class = sorted(classes)[0]
        
        # Convert to binary (1 for positive, 0 for negative)
        y_pred_binary = [1 if p == positive_class else 0 for p in y_pred]
        y_true_binary = [1 if t == positive_class else 0 for t in y_true]
        
        # Calculate confusion matrix
        tp = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 0 and t == 1)
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            },
            "positive_class": positive_class,
            "negative_class": negative_class,
            "class_distribution": {
                "predicted": dict(Counter(y_pred)),
                "actual": dict(Counter(y_true))
            }
        }
    
    def _calculate_multiclass_metrics(
        self,
        y_pred: List[str],
        y_true: List[str],
        classes: List[str]
    ) -> Dict:
        """
        Calculate metrics for multiclass classification
        """
        
        # Overall accuracy
        correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
        accuracy = correct / len(y_pred) if len(y_pred) > 0 else 0
        
        # Per-class metrics
        class_metrics = {}
        
        for cls in classes:
            # Binary view: this class vs all others
            y_pred_binary = [1 if p == cls else 0 for p in y_pred]
            y_true_binary = [1 if t == cls else 0 for t in y_true]
            
            tp = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(y_pred_binary, y_true_binary) if p == 0 and t == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": sum(1 for t in y_true if t == cls)
            }
        
        # Macro averages (unweighted)
        macro_precision = np.mean([m["precision"] for m in class_metrics.values()])
        macro_recall = np.mean([m["recall"] for m in class_metrics.values()])
        macro_f1 = np.mean([m["f1_score"] for m in class_metrics.values()])
        
        # Confusion matrix
        confusion_matrix = {}
        for true_cls in classes:
            confusion_matrix[true_cls] = {}
            for pred_cls in classes:
                count = sum(1 for p, t in zip(y_pred, y_true) if p == pred_cls and t == true_cls)
                confusion_matrix[true_cls][pred_cls] = count
        
        return {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "class_metrics": class_metrics,
            "confusion_matrix": confusion_matrix,
            "class_distribution": {
                "predicted": dict(Counter(y_pred)),
                "actual": dict(Counter(y_true))
            }
        }
    
    def save_metrics(
        self,
        client_id: str,
        model_name: str,
        metrics: Dict,
        period_days: int = 30
    ):
        """
        Save calculated metrics to database
        """
        
        period_start = datetime.utcnow() - timedelta(days=period_days)
        period_end = datetime.utcnow()
        
        perf_metric = PerformanceMetrics(
            client_id=client_id,
            model_name=model_name,
            period_start=period_start,
            period_end=period_end,
            total_predictions=metrics.get('total_predictions', 0),
            predictions_with_outcomes=metrics.get('predictions_with_outcomes', 0),
            accuracy=metrics.get('accuracy'),
            precision_score=metrics.get('precision') or metrics.get('macro_precision'),
            recall_score=metrics.get('recall') or metrics.get('macro_recall'),
            f1_score=metrics.get('f1_score') or metrics.get('macro_f1'),
            confusion_matrix=metrics.get('confusion_matrix'),
            class_metrics=metrics.get('class_metrics')
        )
        
        self.db.add(perf_metric)
        self.db.commit()
        
        return perf_metric
    
    def check_for_degradation(
        self,
        client_id: str,
        model_name: str,
        current_metrics: Dict,
        thresholds: Dict = None
    ) -> List[Dict]:
        """
        Check if performance has degraded below thresholds
        
        Args:
            client_id: Client ID
            model_name: Model name
            current_metrics: Current performance metrics
            thresholds: Dict of metric_name: threshold_value
        
        Returns:
            List of alerts triggered
        """
        
        if thresholds is None:
            thresholds = {
                'accuracy': 0.80,
                'precision': 0.75,
                'recall': 0.75,
                'f1_score': 0.75
            }
        
        alerts = []
        
        for metric_name, threshold in thresholds.items():
            current_value = current_metrics.get(metric_name)
            
            if current_value is None:
                continue
            
            if current_value < threshold:
                # Get previous value for comparison
                prev_metric = self.db.query(PerformanceMetrics).filter(
                    PerformanceMetrics.client_id == client_id,
                    PerformanceMetrics.model_name == model_name
                ).order_by(PerformanceMetrics.calculated_at.desc()).offset(1).first()
                
                previous_value = None
                if prev_metric:
                    previous_value = getattr(prev_metric, f"{metric_name.replace('_score', '_score')}", None)
                    if metric_name == 'precision':
                        previous_value = prev_metric.precision_score
                    elif metric_name == 'recall':
                        previous_value = prev_metric.recall_score
                
                # Determine severity
                drop = threshold - current_value
                if drop > 0.15:
                    severity = 'critical'
                elif drop > 0.10:
                    severity = 'high'
                elif drop > 0.05:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Create alert
                alert = PerformanceAlert(
                    client_id=client_id,
                    model_name=model_name,
                    alert_type=f'{metric_name}_drop',
                    severity=severity,
                    metric_name=metric_name,
                    current_value=float(current_value),
                    threshold_value=float(threshold),
                    previous_value=float(previous_value) if previous_value else None
                )
                
                self.db.add(alert)
                alerts.append({
                    "metric": metric_name,
                    "current_value": current_value,
                    "threshold": threshold,
                    "previous_value": previous_value,
                    "severity": severity
                })
        
        self.db.commit()
        
        return alerts