import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics import confusion_matrix
import json
import os

DB_PATH = "logs/audit_log.db"
FAIRNESS_REPORT_PATH = "logs/fairness_report.json"

# ══════════════════════════════════════════════════════════
# FAIRNESS METRICS CALCULATION
# ══════════════════════════════════════════════════════════

def calculate_fairness_metrics(df, protected_attr, favorable_outcome=1):
    """
    Calculate comprehensive fairness metrics for a protected attribute.
    
    Args:
        df: DataFrame with columns [protected_attr, 'prediction', 'actual']
        protected_attr: Column name for the protected attribute (e.g., 'age_group')
        favorable_outcome: What counts as favorable (1 = Diabetes detected)
    
    Returns:
        Dictionary of fairness metrics per group
    """
    
    groups = df[protected_attr].unique()
    metrics = {}
    
    for group in groups:
        group_data = df[df[protected_attr] == group]
        
        if len(group_data) == 0:
            continue
        
        # Basic counts
        total = len(group_data)
        predicted_positive = (group_data['prediction'] == favorable_outcome).sum()
        actual_positive = (group_data['actual'] == favorable_outcome).sum()
        
        # Confusion matrix components
        tp = ((group_data['prediction'] == favorable_outcome) & 
              (group_data['actual'] == favorable_outcome)).sum()
        fp = ((group_data['prediction'] == favorable_outcome) & 
              (group_data['actual'] != favorable_outcome)).sum()
        tn = ((group_data['prediction'] != favorable_outcome) & 
              (group_data['actual'] != favorable_outcome)).sum()
        fn = ((group_data['prediction'] != favorable_outcome) & 
              (group_data['actual'] == favorable_outcome)).sum()
        
        # Core fairness metrics
        selection_rate = predicted_positive / total if total > 0 else 0
        true_positive_rate = tp / actual_positive if actual_positive > 0 else 0  # Recall / TPR
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR
        precision = tp / predicted_positive if predicted_positive > 0 else 0
        
        metrics[str(group)] = {
            "total_samples": int(total),
            "predicted_positive": int(predicted_positive),
            "actual_positive": int(actual_positive),
            "selection_rate": round(float(selection_rate), 4),
            "true_positive_rate": round(float(true_positive_rate), 4),
            "false_positive_rate": round(float(false_positive_rate), 4),
            "precision": round(float(precision), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    
    return metrics


def calculate_fairness_disparities(group_metrics):
    """
    Calculate disparity ratios between groups.
    
    Fairness criteria:
    - Demographic Parity: Selection rates should be similar across groups
    - Equal Opportunity: TPR should be similar across groups
    - Equalized Odds: Both TPR and FPR should be similar across groups
    """
    
    groups = list(group_metrics.keys())
    
    if len(groups) < 2:
        return {"error": "Need at least 2 groups for disparity analysis"}
    
    # Get reference group (typically the privileged/majority group)
    # For age, we'll use the largest group as reference
    reference_group = max(groups, key=lambda g: group_metrics[g]["total_samples"])
    ref_metrics = group_metrics[reference_group]
    
    disparities = {
        "reference_group": reference_group,
        "groups": {}
    }
    
    for group in groups:
        if group == reference_group:
            continue
        
        group_met = group_metrics[group]
        
        # Disparate Impact Ratio (80% rule)
        selection_ratio = (group_met["selection_rate"] / ref_metrics["selection_rate"] 
                          if ref_metrics["selection_rate"] > 0 else 0)
        
        # Equal Opportunity difference
        tpr_diff = abs(group_met["true_positive_rate"] - ref_metrics["true_positive_rate"])
        
        # Equalized Odds (average of TPR and FPR differences)
        fpr_diff = abs(group_met["false_positive_rate"] - ref_metrics["false_positive_rate"])
        eq_odds_diff = (tpr_diff + fpr_diff) / 2
        
        disparities["groups"][group] = {
            "disparate_impact_ratio": round(float(selection_ratio), 4),
            "passes_80_rule": 0.8 <= selection_ratio <= 1.25,
            "equal_opportunity_diff": round(float(tpr_diff), 4),
            "equalized_odds_diff": round(float(eq_odds_diff), 4),
            "selection_rate_diff": round(float(group_met["selection_rate"] - 
                                               ref_metrics["selection_rate"]), 4)
        }
    
    return disparities


# ══════════════════════════════════════════════════════════
# BIAS DETECTION FROM AUDIT LOG
# ══════════════════════════════════════════════════════════

def analyze_bias_from_audit_log():
    """
    Load predictions from audit log and analyze for bias across age groups.
    In a real system, you'd also analyze by gender, race, etc.
    """
    
    if not os.path.exists(DB_PATH):
        return {"error": "No audit log found"}
    
    # Load audit log
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT age, glucose, bmi, prediction, prediction_label
        FROM audit_log
    """, conn)
    conn.close()
    
    if len(df) < 10:
        return {"error": "Not enough predictions for bias analysis (need 10+)"}
    
    # Create age groups (protected attribute)
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 40, 50, 100],
        labels=['18-30', '31-40', '41-50', '50+']
    )
    
    # For demonstration, we'll create synthetic "actual" outcomes
    # In production, you'd have ground truth labels
    # Here we simulate by adding some noise to predictions
    np.random.seed(42)
    df['actual'] = df['prediction'].copy()
    
    # Introduce realistic error pattern (model slightly worse on older patients)
    for idx in df[df['age_group'] == '50+'].index:
        if np.random.random() < 0.15:  # 15% error rate for 50+
            df.loc[idx, 'actual'] = 1 - df.loc[idx, 'actual']
    
    for idx in df[df['age_group'] != '50+'].index:
        if np.random.random() < 0.08:  # 8% error rate for younger
            df.loc[idx, 'actual'] = 1 - df.loc[idx, 'actual']
    
    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(df, 'age_group', favorable_outcome=1)
    disparities = calculate_fairness_disparities(fairness_metrics)
    
    # Generate bias report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_predictions": len(df),
        "protected_attribute": "age_group",
        "group_metrics": fairness_metrics,
        "disparities": disparities,
        "overall_assessment": assess_overall_fairness(disparities)
    }
    
    # Save report
    os.makedirs("logs", exist_ok=True)
    with open(FAIRNESS_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("  FAIRNESS & BIAS ANALYSIS REPORT")
    print("="*60)
    print(f"\nTotal Predictions Analyzed: {len(df)}")
    print(f"Protected Attribute: Age Group")
    print(f"\n{'Age Group':<15} {'Total':<10} {'Selection Rate':<18} {'TPR':<10} {'FPR':<10}")
    print("-" * 65)
    
    for group, metrics in fairness_metrics.items():
        print(f"{group:<15} {metrics['total_samples']:<10} "
              f"{metrics['selection_rate']:<18.2%} "
              f"{metrics['true_positive_rate']:<10.2%} "
              f"{metrics['false_positive_rate']:<10.2%}")
    
    print("\n" + "="*60)
    print("BIAS DETECTION RESULTS:")
    print("="*60)
    
    if "groups" in disparities:
        for group, disp in disparities["groups"].items():
            status = "✅ PASS" if disp["passes_80_rule"] else "⚠️ FAIL"
            print(f"\n{group} vs {disparities['reference_group']}:")
            print(f"  Disparate Impact Ratio: {disp['disparate_impact_ratio']:.2f} {status}")
            print(f"  Equal Opportunity Diff: {disp['equal_opportunity_diff']:.4f}")
            print(f"  Equalized Odds Diff:    {disp['equalized_odds_diff']:.4f}")
    
    assessment = report["overall_assessment"]
    print(f"\n{'='*60}")
    print(f"Overall Fairness: {assessment['status']}")
    print(f"Recommendation: {assessment['recommendation']}")
    print(f"{'='*60}\n")
    
    return report


def assess_overall_fairness(disparities):
    """Provide overall fairness assessment and recommendations"""
    
    if "error" in disparities or "groups" not in disparities:
        return {
            "status": "INSUFFICIENT_DATA",
            "recommendation": "Need more predictions for comprehensive analysis"
        }
    
    issues = []
    passes_80_rule = all(d["passes_80_rule"] for d in disparities["groups"].values())
    
    max_tpr_diff = max(d["equal_opportunity_diff"] for d in disparities["groups"].values())
    max_eq_odds_diff = max(d["equalized_odds_diff"] for d in disparities["groups"].values())
    
    if not passes_80_rule:
        issues.append("Fails 80% rule for disparate impact")
    
    if max_tpr_diff > 0.1:
        issues.append(f"High TPR disparity ({max_tpr_diff:.2%})")
    
    if max_eq_odds_diff > 0.1:
        issues.append(f"High equalized odds disparity ({max_eq_odds_diff:.2%})")
    
    if not issues:
        status = "FAIR"
        recommendation = "Model shows acceptable fairness across age groups. Continue monitoring."
    elif len(issues) == 1:
        status = "MODERATE_BIAS"
        recommendation = f"Moderate bias detected: {issues[0]}. Consider rebalancing training data or using fairness constraints."
    else:
        status = "SIGNIFICANT_BIAS"
        recommendation = f"Multiple fairness issues: {'; '.join(issues)}. Recommend model retraining with fairness interventions."
    
    return {
        "status": status,
        "issues": issues,
        "recommendation": recommendation,
        "passes_80_rule": passes_80_rule,
        "max_tpr_disparity": round(float(max_tpr_diff), 4),
        "max_equalized_odds": round(float(max_eq_odds_diff), 4)
    }


def load_fairness_report():
    """Load the latest fairness report"""
    if not os.path.exists(FAIRNESS_REPORT_PATH):
        return None
    
    with open(FAIRNESS_REPORT_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
# MITIGATION RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

def generate_mitigation_strategies(report):
    """Generate actionable mitigation strategies based on bias analysis"""
    
    if not report or "overall_assessment" not in report:
        return []
    
    assessment = report["overall_assessment"]
    status = assessment["status"]
    
    strategies = []
    
    if status == "FAIR":
        strategies.append({
            "priority": "LOW",
            "strategy": "Monitoring",
            "description": "Continue monitoring fairness metrics on new predictions",
            "action": "Set up automated fairness checks on weekly batches"
        })
    
    if not assessment.get("passes_80_rule", True):
        strategies.append({
            "priority": "HIGH",
            "strategy": "Rebalancing",
            "description": "Training data may be imbalanced across age groups",
            "action": "Collect more data from underrepresented age groups or use SMOTE for synthetic balancing"
        })
    
    if assessment.get("max_tpr_disparity", 0) > 0.1:
        strategies.append({
            "priority": "HIGH",
            "strategy": "Calibration",
            "description": "Model performs differently across age groups",
            "action": "Apply group-specific decision thresholds or use post-processing calibration"
        })
    
    if assessment.get("max_equalized_odds", 0) > 0.1:
        strategies.append({
            "priority": "MEDIUM",
            "strategy": "Fairness Constraints",
            "description": "Both false positives and false negatives differ across groups",
            "action": "Retrain model with fairness constraints (e.g., using Fairlearn library)"
        })
    
    if status == "SIGNIFICANT_BIAS":
        strategies.append({
            "priority": "CRITICAL",
            "strategy": "Model Redesign",
            "description": "Significant bias detected across multiple fairness dimensions",
            "action": "Consider ensemble approach with fairness-aware models or algorithmic interventions"
        })
    
    return strategies


if __name__ == "__main__":
    # Run bias analysis
    report = analyze_bias_from_audit_log()
    
    if report and "error" not in report:
        print("\n📋 MITIGATION STRATEGIES:")
        print("="*60)
        strategies = generate_mitigation_strategies(report)
        for i, strat in enumerate(strategies, 1):
            print(f"\n{i}. [{strat['priority']}] {strat['strategy']}")
            print(f"   {strat['description']}")
            print(f"   → Action: {strat['action']}")