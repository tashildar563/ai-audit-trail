"""
Regulatory Compliance Database
Defines requirements for different regulations
"""

REGULATIONS = {
    "EU_AI_ACT_HIGH_RISK": {
        "name": "EU AI Act (High-Risk Systems)",
        "jurisdiction": "European Union",
        "status": "active",
        "effective_date": "2025-08-02",
        "description": "Requirements for high-risk AI systems under EU AI Act",
        "applies_to": ["hiring", "credit_scoring", "law_enforcement", "critical_infrastructure"],
        "requirements": [
            {
                "id": "eu_req_1",
                "name": "Bias Monitoring",
                "description": "Continuous monitoring for bias across protected attributes",
                "category": "fairness",
                "weight": 20,
                "check_function": "check_has_fairness_monitoring",
                "threshold": {
                    "disparate_impact_min": 0.8,
                    "audit_frequency_days": 90
                },
                "mandatory": True
            },
            {
                "id": "eu_req_2",
                "name": "Technical Documentation",
                "description": "Documented model architecture, training data, and performance metrics",
                "category": "documentation",
                "weight": 15,
                "check_function": "check_has_documentation",
                "mandatory": True
            },
            {
                "id": "eu_req_3",
                "name": "Accuracy Requirements",
                "description": "Defined and maintained accuracy thresholds",
                "category": "performance",
                "weight": 15,
                "check_function": "check_tracks_performance",
                "mandatory": True
            },
            {
                "id": "eu_req_4",
                "name": "Human Oversight",
                "description": "Human review capability for high-stakes decisions",
                "category": "governance",
                "weight": 15,
                "check_function": "check_has_human_oversight",
                "mandatory": True
            },
            {
                "id": "eu_req_5",
                "name": "Transparency",
                "description": "Users informed that AI is being used",
                "category": "transparency",
                "weight": 10,
                "check_function": "check_has_transparency_notice",
                "mandatory": True
            },
            {
                "id": "eu_req_6",
                "name": "Post-Market Monitoring",
                "description": "Continuous monitoring in production environment",
                "category": "monitoring",
                "weight": 10,
                "check_function": "check_has_production_monitoring",
                "mandatory": False
            },
            {
                "id": "eu_req_7",
                "name": "Data Governance",
                "description": "Training data quality and representativeness documented",
                "category": "data",
                "weight": 10,
                "check_function": "check_data_governance",
                "mandatory": False
            },
            {
                "id": "eu_req_8",
                "name": "Risk Assessment",
                "description": "Risk identification and mitigation strategies",
                "category": "risk",
                "weight": 5,
                "check_function": "check_risk_assessment",
                "mandatory": False
            }
        ],
        "penalties": {
            "max_fine_euros": 30000000,
            "or_percentage_revenue": 0.06,
            "description": "Up to €30M or 6% of global annual turnover, whichever is higher"
        }
    },
    
    "NYC_LAW_144": {
        "name": "NYC Local Law 144 (Automated Employment Decision Tools)",
        "jurisdiction": "New York City, USA",
        "status": "active",
        "effective_date": "2023-07-05",
        "description": "Requirements for AI tools used in hiring and promotion in NYC",
        "applies_to": ["hiring", "promotion", "employment_decisions"],
        "requirements": [
            {
                "id": "nyc_req_1",
                "name": "Annual Bias Audit",
                "description": "Independent bias audit conducted at least annually",
                "category": "fairness",
                "weight": 30,
                "check_function": "check_has_annual_bias_audit",
                "threshold": {
                    "max_age_days": 365,
                    "must_be_independent": True
                },
                "mandatory": True
            },
            {
                "id": "nyc_req_2",
                "name": "Impact Ratio Calculation",
                "description": "Calculate selection rates by race and gender",
                "category": "fairness",
                "weight": 25,
                "check_function": "check_calculates_impact_ratios",
                "threshold": {
                    "required_attributes": ["race", "gender"],
                    "min_impact_ratio": 0.8
                },
                "mandatory": True
            },
            {
                "id": "nyc_req_3",
                "name": "Public Disclosure",
                "description": "Publish audit results on company website",
                "category": "transparency",
                "weight": 20,
                "check_function": "check_public_disclosure",
                "mandatory": True
            },
            {
                "id": "nyc_req_4",
                "name": "Candidate Notification",
                "description": "Notify candidates that AI tool is used",
                "category": "transparency",
                "weight": 15,
                "check_function": "check_candidate_notification",
                "mandatory": True
            },
            {
                "id": "nyc_req_5",
                "name": "Alternative Process",
                "description": "Provide alternative selection process upon request",
                "category": "governance",
                "weight": 10,
                "check_function": "check_alternative_process",
                "mandatory": True
            }
        ],
        "penalties": {
            "per_violation_min": 500,
            "per_violation_max": 1500,
            "description": "$500-$1,500 per violation (can accumulate quickly)"
        }
    }
}


def get_regulation(regulation_id):
    """Get a specific regulation by ID"""
    return REGULATIONS.get(regulation_id)


def get_all_regulations():
    """Get all available regulations"""
    return REGULATIONS


def get_regulations_by_use_case(use_case):
    """
    Get applicable regulations for a specific use case
    
    Args:
        use_case: e.g., "hiring", "credit_scoring", "healthcare"
    
    Returns:
        List of applicable regulations
    """
    applicable = []
    for reg_id, regulation in REGULATIONS.items():
        if use_case.lower() in [uc.lower() for uc in regulation["applies_to"]]:
            applicable.append({
                "id": reg_id,
                "name": regulation["name"],
                "jurisdiction": regulation["jurisdiction"]
            })
    return applicable