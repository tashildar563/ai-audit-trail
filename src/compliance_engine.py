"""
Compliance Engine
Orchestrates compliance checks and generates reports
"""

from sqlalchemy.orm import Session
from src.regulations_db import get_regulation, get_regulations_by_use_case
from src.compliance_checks import get_check_function
from datetime import datetime


class ComplianceEngine:
    """
    Main compliance engine that runs checks and generates reports
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_compliance(
        self, 
        client_id: str, 
        model_name: str, 
        regulation_ids: list,
        use_case: str = None
    ) -> dict:
        """
        Run compliance check for specified regulations
        
        Args:
            client_id: Client ID
            model_name: Model name to check
            regulation_ids: List of regulation IDs to check against
            use_case: Optional use case (e.g., "hiring", "credit_scoring")
        
        Returns:
            Complete compliance report
        """
        
        results = {
            "client_id": client_id,
            "model_name": model_name,
            "use_case": use_case,
            "timestamp": datetime.utcnow().isoformat(),
            "regulations_checked": len(regulation_ids),
            "regulation_results": {},
            "overall_score": 0,
            "overall_status": "UNKNOWN",
            "critical_issues": [],
            "recommendations": [],
            "summary": {}
        }
        
        total_score = 0
        total_weight = 0
        all_passed = True
        
        # Check each regulation
        for reg_id in regulation_ids:
            regulation = get_regulation(reg_id)
            
            if not regulation:
                results["regulation_results"][reg_id] = {
                    "error": f"Regulation '{reg_id}' not found"
                }
                continue
            
            # Run checks for this regulation
            reg_result = self._check_regulation(
                client_id, 
                model_name, 
                regulation
            )
            
            results["regulation_results"][reg_id] = reg_result
            
            # Aggregate scores
            total_score += reg_result["score"] * reg_result["total_weight"]
            total_weight += reg_result["total_weight"]
            
            if not reg_result["passed"]:
                all_passed = False
            
            # Collect critical issues
            for req_result in reg_result["requirement_results"]:
                if req_result["mandatory"] and not req_result["passed"]:
                    results["critical_issues"].append({
                        "regulation": regulation["name"],
                        "requirement": req_result["name"],
                        "issue": req_result["evidence"],
                        "recommendation": req_result["recommendation"]
                    })
            
            # Collect recommendations
            for req_result in reg_result["requirement_results"]:
                if req_result["recommendation"]:
                    results["recommendations"].append({
                        "regulation": regulation["name"],
                        "requirement": req_result["name"],
                        "priority": "CRITICAL" if req_result["mandatory"] and not req_result["passed"] else "MEDIUM",
                        "recommendation": req_result["recommendation"]
                    })
        
        # Calculate overall score
        if total_weight > 0:
            results["overall_score"] = round(total_score / total_weight, 1)
        else:
            results["overall_score"] = 0
        
        # Determine overall status
        score = results["overall_score"]
        if score >= 90:
            results["overall_status"] = "COMPLIANT"
        elif score >= 70:
            results["overall_status"] = "MOSTLY_COMPLIANT"
        elif score >= 50:
            results["overall_status"] = "NEEDS_IMPROVEMENT"
        else:
            results["overall_status"] = "NON_COMPLIANT"
        
        # Generate summary
        results["summary"] = {
            "total_requirements_checked": sum(
                len(r["requirement_results"]) 
                for r in results["regulation_results"].values() 
                if "requirement_results" in r
            ),
            "requirements_passed": sum(
                sum(1 for req in r["requirement_results"] if req["passed"])
                for r in results["regulation_results"].values()
                if "requirement_results" in r
            ),
            "critical_issues_count": len(results["critical_issues"]),
            "regulations_passed": sum(
                1 for r in results["regulation_results"].values() 
                if r.get("passed", False)
            ),
            "regulations_failed": sum(
                1 for r in results["regulation_results"].values() 
                if not r.get("passed", True)
            )
        }
        
        # Sort recommendations by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        results["recommendations"].sort(
            key=lambda x: priority_order.get(x["priority"], 99)
        )
        
        return results
    
    def _check_regulation(
        self, 
        client_id: str, 
        model_name: str, 
        regulation: dict
    ) -> dict:
        """
        Check compliance for a single regulation
        
        Args:
            client_id: Client ID
            model_name: Model name
            regulation: Regulation dict from regulations_db
        
        Returns:
            Regulation compliance result
        """
        
        result = {
            "regulation_name": regulation["name"],
            "jurisdiction": regulation["jurisdiction"],
            "requirement_results": [],
            "score": 0,
            "total_weight": 0,
            "passed": True,
            "mandatory_failed": []
        }
        
        total_score = 0
        total_weight = 0
        
        # Run each requirement check
        for requirement in regulation["requirements"]:
            req_result = self._check_requirement(
                client_id,
                model_name,
                requirement
            )
            
            result["requirement_results"].append(req_result)
            
            # Weight the score
            weighted_score = req_result["score"] * requirement["weight"]
            total_score += weighted_score
            total_weight += requirement["weight"]
            
            # Track mandatory failures
            if requirement["mandatory"] and not req_result["passed"]:
                result["passed"] = False
                result["mandatory_failed"].append(requirement["name"])
        
        # Calculate overall score for this regulation
        if total_weight > 0:
            result["score"] = round(total_score / total_weight, 1)
        else:
            result["score"] = 0
        
        result["total_weight"] = total_weight
        
        return result
    
    def _check_requirement(
        self,
        client_id: str,
        model_name: str,
        requirement: dict
    ) -> dict:
        """
        Check a single requirement
        
        Args:
            client_id: Client ID
            model_name: Model name
            requirement: Requirement dict
        
        Returns:
            Requirement check result
        """
        
        # Get the check function
        check_func = get_check_function(requirement["check_function"])
        
        if not check_func:
            return {
                "id": requirement["id"],
                "name": requirement["name"],
                "description": requirement["description"],
                "category": requirement.get("category", "other"),
                "weight": requirement["weight"],
                "mandatory": requirement["mandatory"],
                "passed": False,
                "score": 0,
                "evidence": f"Check function '{requirement['check_function']}' not implemented",
                "details": {},
                "recommendation": "Contact support - check function missing"
            }
        
        # Run the check
        try:
            check_result = check_func(self.db, client_id, model_name)
            
            return {
                "id": requirement["id"],
                "name": requirement["name"],
                "description": requirement["description"],
                "category": requirement.get("category", "other"),
                "weight": requirement["weight"],
                "mandatory": requirement["mandatory"],
                "passed": check_result["passed"],
                "score": check_result["score"],
                "evidence": check_result["evidence"],
                "details": check_result.get("details", {}),
                "recommendation": check_result.get("recommendation")
            }
        
        except Exception as e:
            return {
                "id": requirement["id"],
                "name": requirement["name"],
                "description": requirement["description"],
                "category": requirement.get("category", "other"),
                "weight": requirement["weight"],
                "mandatory": requirement["mandatory"],
                "passed": False,
                "score": 0,
                "evidence": f"Error running check: {str(e)}",
                "details": {},
                "recommendation": "Contact support - check failed"
            }
    
    def get_applicable_regulations(self, use_case: str) -> list:
        """
        Get list of regulations that apply to a use case
        
        Args:
            use_case: Use case (e.g., "hiring", "credit_scoring")
        
        Returns:
            List of applicable regulation IDs
        """
        return get_regulations_by_use_case(use_case)


def generate_compliance_summary(compliance_report: dict) -> str:
    """
    Generate a human-readable summary of compliance report
    
    Args:
        compliance_report: Output from ComplianceEngine.check_compliance()
    
    Returns:
        Human-readable summary string
    """
    
    status_emoji = {
        "COMPLIANT": "✅",
        "MOSTLY_COMPLIANT": "⚠️",
        "NEEDS_IMPROVEMENT": "⚠️",
        "NON_COMPLIANT": "❌"
    }
    
    emoji = status_emoji.get(compliance_report["overall_status"], "❓")
    
    summary = f"""
{emoji} COMPLIANCE REPORT SUMMARY
{'='*60}

Model: {compliance_report['model_name']}
Overall Status: {compliance_report['overall_status']}
Overall Score: {compliance_report['overall_score']}/100

Regulations Checked: {compliance_report['regulations_checked']}
Requirements Checked: {compliance_report['summary']['total_requirements_checked']}
Requirements Passed: {compliance_report['summary']['requirements_passed']}
Critical Issues: {compliance_report['summary']['critical_issues_count']}

"""
    
    # Critical issues
    if compliance_report["critical_issues"]:
        summary += "\n🚨 CRITICAL ISSUES:\n"
        for i, issue in enumerate(compliance_report["critical_issues"][:5], 1):
            summary += f"\n{i}. {issue['requirement']} ({issue['regulation']})\n"
            summary += f"   Issue: {issue['issue']}\n"
            summary += f"   Action: {issue['recommendation']}\n"
    
    # Top recommendations
    if compliance_report["recommendations"]:
        summary += "\n💡 TOP RECOMMENDATIONS:\n"
        for i, rec in enumerate(compliance_report["recommendations"][:5], 1):
            priority_emoji = "🔴" if rec["priority"] == "CRITICAL" else "🟡"
            summary += f"\n{i}. {priority_emoji} {rec['requirement']}\n"
            summary += f"   {rec['recommendation']}\n"
    
    summary += "\n" + "="*60
    
    return summary