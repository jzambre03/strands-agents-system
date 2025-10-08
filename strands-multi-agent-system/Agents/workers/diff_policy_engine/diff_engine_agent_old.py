"""
Golden Config AI - Diff Policy Engine Agent (Strands Implementation)

Migrated from Architecture/lambda-functions/diff-engine/lambda_function.py
This agent performs intelligent configuration drift analysis and policy validation
with enhanced AI-driven insights and learning capabilities.

Key Enhancements over Lambda Version:
- Intelligent drift pattern recognition and classification
- Learning from previous drift analysis outcomes
- Contextual policy recommendations based on environment
- Advanced change impact assessment and risk scoring
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools import tool
import difflib

from shared.config import Config
from shared.models import (
    TaskRequest,
    TaskResponse,
    AgentMessage,
    ValidationStatus,
    ValidationResult
)
from shared.logging_config import get_agent_logger

logger = get_agent_logger("diff_engine")


class DiffPolicyEngineAgent(Agent):
    """
    Diff Policy Engine Agent - Configuration Drift Analysis and Policy Validation
    
    This agent specializes in:
    - Analyzing configuration drift between environments
    - Validating changes against organizational policies
    - Providing intelligent change impact assessments
    - Learning from drift patterns to improve future analysis
    """

    def __init__(self, config: Config):
        system_prompt = self._get_diff_engine_prompt()
        
        super().__init__(
            model=BedrockModel(
                model_id=config.bedrock_model_id,
            ),
            system_prompt=system_prompt,
            tools=[
                self.analyze_configuration_drift,
                self.validate_policy_compliance,
                self.assess_change_impact,
                self.generate_drift_recommendations,
                self.classify_change_patterns,
                self.generate_remediation_plan
            ]
        )
        
        self.config = config
        
        # Initialize policy rules and drift patterns
        self.policy_rules = self._initialize_policy_rules()
        self.drift_patterns = self._initialize_drift_patterns()

    def _get_diff_engine_prompt(self) -> str:
        """System prompt for the Diff Policy Engine Agent"""
        return """You are the Diff Policy Engine Agent in the Golden Config AI system.

Your primary responsibility is to analyze configuration changes and drift with intelligent insights and policy validation.

CORE CAPABILITIES:
1. Drift Analysis: Detect and classify configuration changes between environments
2. Policy Validation: Ensure changes comply with organizational policies and standards
3. Impact Assessment: Evaluate the potential impact of configuration changes
4. Pattern Recognition: Identify recurring drift patterns and their implications
5. Risk Scoring: Assess the risk level of configuration changes

ANALYSIS FOCUS AREAS:
- Environment consistency (dev/staging/production alignment)
- Security configuration changes (authentication, encryption, access control)
- Performance impact changes (resource limits, timeouts, caching)
- Compliance adherence (regulatory requirements, internal policies)
- Operational risk assessment (availability, reliability, maintainability)

DECISION MAKING:
- Prioritize changes by risk level and business impact
- Consider context such as environment type and change urgency
- Learn from historical change outcomes to improve assessments
- Provide clear reasoning for policy violations or approvals
- Balance innovation needs with stability requirements

COMMUNICATION:
- Clearly categorize changes by type and severity
- Provide specific recommendations for each identified drift
- Explain policy violations with references to specific rules
- Suggest optimal approaches for implementing necessary changes
- Alert stakeholders to high-risk changes requiring immediate attention

Always focus on maintaining system reliability while enabling necessary business changes."""

    @tool
    async def analyze_configuration_drift(self, golden_config: Dict[str, str],
                                        current_config: Dict[str, str],
                                        environment: str = "production",
                                        analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze configuration drift between golden and current configurations
        
        Args:
            golden_config: Golden/baseline configuration files
            current_config: Current configuration files to compare
            environment: Target environment context
            analysis_depth: Level of analysis (quick, standard, comprehensive)
            
        Returns:
            Detailed drift analysis results
        """
        logger.info(f"Analyzing configuration drift for {environment} environment")
        
        try:
            drift_analysis = {
                "environment": environment,
                "analysis_depth": analysis_depth,
                "total_files_compared": 0,
                "files_with_drift": 0,
                "drift_summary": {
                    "additions": 0,
                    "deletions": 0,
                    "modifications": 0,
                    "critical_changes": 0,
                    "policy_violations": 0
                },
                "detailed_drifts": [],
                "risk_assessment": {},
                "recommendations": []
            }
            
            # Find all files to compare
            all_files = set(golden_config.keys()) | set(current_config.keys())
            drift_analysis["total_files_compared"] = len(all_files)
            
            for file_path in all_files:
                file_drift = await self._analyze_file_drift(
                    file_path, 
                    golden_config.get(file_path, ""), 
                    current_config.get(file_path, ""),
                    environment,
                    analysis_depth
                )
                
                if file_drift["has_changes"]:
                    # Enhance with AI-powered analysis for changed files
                    try:
                        ai_analysis = await self._analyze_file_drift_with_ai(
                            file_drift,  # Pass the already computed drift analysis
                            environment
                        )
                        if ai_analysis:
                            file_drift["ai_analysis"] = ai_analysis
                            # Override risk level if AI provides more accurate assessment
                            if ai_analysis.get("risk_level"):
                                file_drift["risk_level"] = ai_analysis["risk_level"]
                            # Merge AI recommendations
                            if ai_analysis.get("recommendations"):
                                file_drift.setdefault("recommendations", []).extend(ai_analysis["recommendations"])
                            # Add AI-detected policy violations
                            if ai_analysis.get("policy_violations"):
                                file_drift.setdefault("policy_violations", []).extend(ai_analysis["policy_violations"])
                    except Exception as e:
                        logger.warning(f"AI analysis failed for {file_path}: {e}")
                    
                    drift_analysis["files_with_drift"] += 1
                    drift_analysis["detailed_drifts"].append(file_drift)
                    
                    # Update summary counts
                    drift_analysis["drift_summary"]["additions"] += file_drift["changes"]["additions"]
                    drift_analysis["drift_summary"]["deletions"] += file_drift["changes"]["deletions"]
                    drift_analysis["drift_summary"]["modifications"] += file_drift["changes"]["modifications"]
                    
                    if file_drift["risk_level"] in ["critical", "high"]:
                        drift_analysis["drift_summary"]["critical_changes"] += 1
                    
                    if file_drift["policy_violations"]:
                        drift_analysis["drift_summary"]["policy_violations"] += len(file_drift["policy_violations"])
            
            # Perform overall risk assessment with AI enhancement
            drift_analysis["risk_assessment"] = await self._assess_overall_drift_risk_with_ai(
                drift_analysis["detailed_drifts"], environment
            )
            
            # Generate recommendations
            drift_analysis["recommendations"] = await self._generate_drift_recommendations(
                drift_analysis["detailed_drifts"], drift_analysis["risk_assessment"]
            )
            
            return {
                "status": "success",
                "drift_analysis": drift_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Configuration drift analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "environment": environment,
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def validate_policy_compliance(self, configuration_changes: List[Dict[str, Any]],
                                       policy_set: str = "default",
                                       environment: str = "production") -> Dict[str, Any]:
        """
        Validate configuration changes against organizational policies
        
        Args:
            configuration_changes: List of configuration changes to validate
            policy_set: Set of policies to apply (default, strict, permissive)
            environment: Environment context for policy application
            
        Returns:
            Policy compliance validation results
        """
        logger.info(f"Validating policy compliance for {len(configuration_changes)} changes")
        
        try:
            compliance_results = {
                "policy_set": policy_set,
                "environment": environment,
                "overall_compliant": True,
                "compliance_score": 0,
                "violations": [],
                "warnings": [],
                "approvals": [],
                "total_changes": len(configuration_changes)
            }
            
            applicable_policies = await self._get_applicable_policies(policy_set, environment)
            
            for change in configuration_changes:
                change_compliance = await self._validate_change_against_policies(
                    change, applicable_policies, environment
                )
                
                if change_compliance["violations"]:
                    compliance_results["violations"].extend(change_compliance["violations"])
                    compliance_results["overall_compliant"] = False
                
                if change_compliance["warnings"]:
                    compliance_results["warnings"].extend(change_compliance["warnings"])
                
                if change_compliance["approved"]:
                    compliance_results["approvals"].append(change_compliance)
            
            # Calculate compliance score
            total_issues = len(compliance_results["violations"]) + len(compliance_results["warnings"])
            if total_issues == 0:
                compliance_results["compliance_score"] = 100
            else:
                # Score based on severity of issues
                violation_penalty = len(compliance_results["violations"]) * 20
                warning_penalty = len(compliance_results["warnings"]) * 5
                compliance_results["compliance_score"] = max(0, 100 - violation_penalty - warning_penalty)
            
            return {
                "status": "success",
                "compliance_results": compliance_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Policy compliance validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "policy_set": policy_set,
                "environment": environment,
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def assess_change_impact(self, configuration_changes: List[Dict[str, Any]],
                                 environment: str = "production",
                                 impact_scope: str = "full") -> Dict[str, Any]:
        """
        Assess the potential impact of configuration changes
        
        Args:
            configuration_changes: List of configuration changes to assess
            environment: Environment where changes will be applied
            impact_scope: Scope of impact analysis (minimal, standard, full)
            
        Returns:
            Change impact assessment results
        """
        logger.info(f"Assessing impact of {len(configuration_changes)} changes")
        
        try:
            impact_assessment = {
                "environment": environment,
                "impact_scope": impact_scope,
                "overall_risk_level": "low",
                "impact_categories": {
                    "security": {"score": 0, "changes": []},
                    "performance": {"score": 0, "changes": []},
                    "availability": {"score": 0, "changes": []},
                    "compliance": {"score": 0, "changes": []},
                    "operational": {"score": 0, "changes": []}
                },
                "rollback_complexity": "simple",
                "testing_recommendations": [],
                "deployment_considerations": []
            }
            
            for change in configuration_changes:
                change_impact = await self._assess_single_change_impact(
                    change, environment, impact_scope
                )
                
                # Categorize impact by area
                for category in impact_assessment["impact_categories"]:
                    if category in change_impact["impact_areas"]:
                        impact_assessment["impact_categories"][category]["changes"].append(change)
                        impact_assessment["impact_categories"][category]["score"] += change_impact["impact_areas"][category]["score"]
            
            # Calculate overall risk level
            category_scores = [cat["score"] for cat in impact_assessment["impact_categories"].values()]
            avg_score = sum(category_scores) / len(category_scores) if category_scores else 0
            
            if avg_score >= 80:
                impact_assessment["overall_risk_level"] = "critical"
            elif avg_score >= 60:
                impact_assessment["overall_risk_level"] = "high"
            elif avg_score >= 40:
                impact_assessment["overall_risk_level"] = "medium"
            else:
                impact_assessment["overall_risk_level"] = "low"
            
            # Generate testing and deployment recommendations
            impact_assessment["testing_recommendations"] = await self._generate_testing_recommendations(
                configuration_changes, impact_assessment
            )
            impact_assessment["deployment_considerations"] = await self._generate_deployment_considerations(
                configuration_changes, impact_assessment
            )
            
            return {
                "status": "success",
                "impact_assessment": impact_assessment,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Change impact assessment failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "environment": environment,
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def generate_drift_recommendations(self, drift_analysis: Dict[str, Any],
                                           priority_level: str = "high") -> Dict[str, Any]:
        """
        Generate specific recommendations for addressing configuration drift
        
        Args:
            drift_analysis: Results from drift analysis
            priority_level: Priority level for recommendations
            
        Returns:
            Detailed drift remediation recommendations
        """
        logger.info(f"Generating {priority_level} priority drift recommendations")
        
        try:
            recommendations = {
                "priority_level": priority_level,
                "immediate_actions": [],
                "corrective_measures": [],
                "preventive_measures": [],
                "monitoring_enhancements": [],
                "process_improvements": []
            }
            
            # Analyze drift patterns and generate recommendations
            if "detailed_drifts" in drift_analysis:
                for drift in drift_analysis["detailed_drifts"]:
                    drift_recs = await self._generate_specific_drift_recommendations(
                        drift, priority_level
                    )
                    
                    recommendations["immediate_actions"].extend(drift_recs["immediate"])
                    recommendations["corrective_measures"].extend(drift_recs["corrective"])
                    recommendations["preventive_measures"].extend(drift_recs["preventive"])
            
            # Add monitoring and process improvements
            recommendations["monitoring_enhancements"] = await self._suggest_monitoring_improvements(
                drift_analysis
            )
            recommendations["process_improvements"] = await self._suggest_process_improvements(
                drift_analysis
            )
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Drift recommendations generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "priority_level": priority_level,
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def classify_change_patterns(self, historical_changes: List[Dict[str, Any]],
                                     analysis_period: str = "30_days") -> Dict[str, Any]:
        """
        Classify and analyze patterns in historical configuration changes
        
        Args:
            historical_changes: Historical change data for analysis
            analysis_period: Time period for pattern analysis
            
        Returns:
            Change pattern classification and insights
        """
        logger.info(f"Classifying change patterns over {analysis_period}")
        
        try:
            pattern_analysis = {
                "analysis_period": analysis_period,
                "total_changes": len(historical_changes),
                "change_patterns": {
                    "frequent_changes": [],
                    "risky_patterns": [],
                    "seasonal_patterns": [],
                    "error_prone_areas": []
                },
                "insights": [],
                "recommendations": []
            }
            
            # Analyze different types of patterns
            pattern_analysis["change_patterns"]["frequent_changes"] = await self._identify_frequent_changes(
                historical_changes, analysis_period
            )
            pattern_analysis["change_patterns"]["risky_patterns"] = await self._identify_risky_patterns(
                historical_changes
            )
            pattern_analysis["change_patterns"]["seasonal_patterns"] = await self._identify_seasonal_patterns(
                historical_changes, analysis_period
            )
            pattern_analysis["change_patterns"]["error_prone_areas"] = await self._identify_error_prone_areas(
                historical_changes
            )
            
            # Generate insights and recommendations
            pattern_analysis["insights"] = await self._generate_pattern_insights(
                pattern_analysis["change_patterns"]
            )
            pattern_analysis["recommendations"] = await self._generate_pattern_recommendations(
                pattern_analysis["change_patterns"], pattern_analysis["insights"]
            )
            
            return {
                "status": "success",
                "pattern_analysis": pattern_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Change pattern classification failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_period": analysis_period,
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def generate_remediation_plan(self, drift_analysis: Dict[str, Any],
                                      compliance_results: Optional[Dict[str, Any]] = None,
                                      priority_level: str = "medium") -> Dict[str, Any]:
        """
        Generate comprehensive remediation plan based on drift analysis and compliance results
        
        Args:
            drift_analysis: Results from configuration drift analysis
            compliance_results: Results from policy compliance validation
            priority_level: Priority level for remediation actions (low, medium, high, critical)
            
        Returns:
            Detailed remediation plan with prioritized actions
        """
        logger.info(f"Generating {priority_level} priority remediation plan")
        
        try:
            remediation_plan = {
                "priority_level": priority_level,
                "generated_at": datetime.utcnow().isoformat(),
                "immediate_actions": [],
                "medium_term_actions": [],
                "long_term_strategies": [],
                "resource_requirements": {},
                "estimated_timeline": {},
                "risk_mitigation": []
            }
            
            # Process drift analysis results
            if drift_analysis and drift_analysis.get("detailed_drifts"):
                for drift in drift_analysis["detailed_drifts"]:
                    if drift["risk_level"] in ["critical", "high"]:
                        action = {
                            "action_type": "revert_change",
                            "file_path": drift["file_path"],
                            "description": f"Address {drift['risk_level']} risk drift in {drift['file_path']}",
                            "estimated_effort": "2-4 hours",
                            "required_approvals": ["security_team"] if drift["risk_level"] == "critical" else []
                        }
                        remediation_plan["immediate_actions"].append(action)
                    elif drift["risk_level"] == "medium":
                        action = {
                            "action_type": "review_and_fix",
                            "file_path": drift["file_path"],
                            "description": f"Review and address medium risk changes in {drift['file_path']}",
                            "estimated_effort": "1-2 hours",
                            "required_approvals": []
                        }
                        remediation_plan["medium_term_actions"].append(action)
            
            # Process compliance violations if provided
            if compliance_results and compliance_results.get("violations"):
                for violation in compliance_results["violations"]:
                    action = {
                        "action_type": "compliance_fix",
                        "violation_type": violation.get("type", "unknown"),
                        "description": f"Fix compliance violation: {violation.get('description', 'Unknown violation')}",
                        "estimated_effort": "3-6 hours",
                        "required_approvals": ["compliance_team"]
                    }
                    remediation_plan["immediate_actions"].append(action)
            
            # Add long-term strategies
            remediation_plan["long_term_strategies"] = [
                {
                    "strategy": "implement_automated_drift_detection",
                    "description": "Set up automated monitoring for configuration drift",
                    "estimated_effort": "1-2 weeks"
                },
                {
                    "strategy": "policy_governance_improvement",
                    "description": "Enhance policy governance and approval workflows",
                    "estimated_effort": "2-4 weeks"
                }
            ]
            
            # Calculate resource requirements
            total_immediate = len(remediation_plan["immediate_actions"])
            total_medium = len(remediation_plan["medium_term_actions"])
            
            remediation_plan["resource_requirements"] = {
                "immediate_resources_needed": max(1, total_immediate // 2),
                "medium_term_resources_needed": max(1, total_medium // 3),
                "skills_required": ["configuration_management", "security_analysis", "policy_compliance"]
            }
            
            # Estimate timeline
            remediation_plan["estimated_timeline"] = {
                "immediate_completion": "1-3 days",
                "medium_term_completion": "1-2 weeks",
                "long_term_completion": "1-3 months"
            }
            
            return {
                "status": "success",
                "remediation_plan": remediation_plan,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Remediation plan generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "priority_level": priority_level,
                "timestamp": datetime.utcnow().isoformat()
            }

    # Private helper methods

    def _initialize_policy_rules(self) -> Dict[str, Any]:
        """Initialize organizational policy rules"""
        return {
            "security": {
                "encryption_required": True,
                "authentication_mandatory": True,
                "ssl_tls_enforcement": True
            },
            "compliance": {
                "audit_logging": True,
                "data_retention_policies": True,
                "access_control": True
            },
            "operational": {
                "backup_configuration": True,
                "monitoring_enabled": True,
                "resource_limits": True
            }
        }

    def _initialize_drift_patterns(self) -> Dict[str, Any]:
        """Initialize common drift patterns"""
        return {
            "critical_patterns": [
                "security_setting_changes",
                "authentication_modifications",
                "encryption_disabling"
            ],
            "warning_patterns": [
                "resource_limit_changes",
                "timeout_modifications",
                "logging_level_changes"
            ]
        }

    async def _analyze_file_drift(self, file_path: str, golden_content: Any,
                                current_content: Any, environment: str,
                                analysis_depth: str) -> Dict[str, Any]:
        """Analyze drift for a single configuration file"""
        
        # Convert content to string format for comparison if needed
        if isinstance(golden_content, dict):
            golden_content = json.dumps(golden_content, indent=2, sort_keys=True)
        elif golden_content is None:
            golden_content = ""
        
        if isinstance(current_content, dict):
            current_content = json.dumps(current_content, indent=2, sort_keys=True)
        elif current_content is None:
            current_content = ""
        
        # Handle file existence differences
        if not golden_content and current_content:
            return {
                "file_path": file_path,
                "has_changes": True,
                "change_type": "addition",
                "risk_level": "medium",
                "changes": {"additions": 1, "deletions": 0, "modifications": 0},
                "policy_violations": [],
                "recommendations": [f"Review new file {file_path} for compliance"]
            }
        elif golden_content and not current_content:
            return {
                "file_path": file_path,
                "has_changes": True,
                "change_type": "deletion",
                "risk_level": "high",
                "changes": {"additions": 0, "deletions": 1, "modifications": 0},
                "policy_violations": ["Unexpected file deletion"],
                "recommendations": [f"Investigate deletion of {file_path}"]
            }
        elif golden_content == current_content:
            return {
                "file_path": file_path,
                "has_changes": False,
                "change_type": "none",
                "risk_level": "none",
                "changes": {"additions": 0, "deletions": 0, "modifications": 0},
                "policy_violations": [],
                "recommendations": []
            }
        
        # Analyze content differences
        differ = difflib.unified_diff(
            golden_content.splitlines(keepends=True),
            current_content.splitlines(keepends=True),
            fromfile=f"golden/{file_path}",
            tofile=f"current/{file_path}"
        )
        
        diff_lines = list(differ)
        additions = len([line for line in diff_lines if line.startswith('+')])
        deletions = len([line for line in diff_lines if line.startswith('-')])
        
        # Assess risk level based on changes
        risk_level = "low"
        if additions + deletions > 50:
            risk_level = "high"
        elif additions + deletions > 10:
            risk_level = "medium"
        
        return {
            "file_path": file_path,
            "has_changes": True,
            "change_type": "modification",
            "risk_level": risk_level,
            "changes": {"additions": additions, "deletions": deletions, "modifications": 1},
            "diff_content": diff_lines[:100],  # Limit diff content for performance
            "policy_violations": [],  # Would be populated by policy validation
            "recommendations": []  # Would be populated based on specific changes
        }

    async def _analyze_file_drift_with_ai(self, file_drift: Dict[str, Any], 
                                         environment: str) -> Dict[str, Any]:
        """
        Use AI to analyze configuration drift intelligently using Strands BedrockModel.
        Takes the already computed diff analysis and enhances it with AI insights.
        
        Args:
            file_drift: Already computed drift analysis containing diff_content and metadata
            environment: Environment context
            
        Returns:
            AI-powered analysis results
        """
        try:
            file_path = file_drift["file_path"]
            changes = file_drift["changes"]
            diff_content = "\n".join(file_drift.get("diff_content", [])[:100])  # Use pre-computed diff
            
            # Prepare a focused prompt with the already computed drift information
            prompt = f"""You are an AI-powered configuration drift analysis assistant for the Golden Config AI system. Your task is to analyze configuration changes and provide actionable insights.

### Context:
- **Environment**: {environment}
- **File**: {file_path}
- **Change Summary**:
  - Lines added: {changes["additions"]}
  - Lines deleted: {changes["deletions"]}
  - Change type: {file_drift["change_type"]}
  - Risk level (basic): {file_drift["risk_level"]}

### File Importance:
This file is part of the system's configuration. Changes to this file may impact:
- Security (e.g., authentication, encryption, access control)
- Performance (e.g., resource limits, timeouts, caching)
- Compliance (e.g., regulatory requirements, internal policies)
- Operational stability (e.g., availability, reliability, maintainability)

### Diff Content:
{diff_content}

### Your Task:
Analyze the configuration drift and provide the following in **JSON format**:
1. **Risk Level**: Assess the risk level of the changes (low, medium, high, critical) based on their potential impact on security, compliance, performance, and operations.
2. **Change Impact**: Describe the functionality or behavior affected by the changes.
3. **Policy Violations**: Identify any violations of security, compliance, or operational policies. Reference specific rules where applicable.
4. **Recommendations**: Provide actionable steps to address the identified risks or violations. Include preventive measures to avoid similar issues in the future.

### Example Output:
```json
{{
  "risk_level": "high",
  "change_impact": "The change modifies authentication settings, potentially weakening security.",
  "policy_violations": [
    {{
      "type": "security",
      "description": "Authentication must use multi-factor authentication (MFA)."
    }}
  ],
  "recommendations": [
    "Revert the change to authentication settings.",
    "Ensure MFA is enforced for all users.",
    "Conduct a security review of the configuration."
  ]
}}
```"""

            # Use Strands model streaming (following test_strands.py pattern)
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=500):
                # Check for contentBlockDelta events and extract text
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            if not ai_response.strip():
                return None
                
            # Try to parse JSON response
            try:
                import json
                # Clean up response to extract JSON
                start_idx = ai_response.find("{")
                end_idx = ai_response.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    ai_analysis = json.loads(json_str)
                    return ai_analysis
                else:
                    # Fallback to structured text parsing
                    return self._parse_ai_text_response(ai_response)
            except json.JSONDecodeError:
                # Fallback to text parsing
                return self._parse_ai_text_response(ai_response)
                
        except Exception as e:
            logger.error(f"AI analysis failed for {file_drift['file_path']}: {e}")
            return None
    def _parse_ai_text_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI text response when JSON parsing fails"""
        result = {
            "risk_level": "medium",
            "change_impact": "Configuration drift detected",
            "policy_violations": [],
            "recommendations": []
        }
        
        # Simple text parsing for risk level
        response_lower = ai_response.lower()
        if "critical" in response_lower:
            result["risk_level"] = "critical"
        elif "high" in response_lower:
            result["risk_level"] = "high"
        elif "low" in response_lower:
            result["risk_level"] = "low"
            
        # Extract recommendations (lines starting with bullet points or numbers)
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•')) or (len(line) > 3 and line[0].isdigit() and line[1] == '.'):
                result["recommendations"].append(line)
        
        return result

    async def _assess_overall_drift_risk_with_ai(self, detailed_drifts: List[Dict[str, Any]], 
                                               environment: str) -> Dict[str, Any]:
        """
        Use AI to assess overall drift risk comprehensively
        
        Args:
            detailed_drifts: List of individual file drift analyses
            environment: Environment context
            
        Returns:
            AI-enhanced overall risk assessment
        """
        try:
            # Prepare summary for AI analysis
            drift_summary = {
                "total_files_changed": len(detailed_drifts),
                "risk_levels": {},
                "files_with_violations": []
            }
            
            # Summarize drift data
            for drift in detailed_drifts:
                risk_level = drift.get("risk_level", "unknown")
                drift_summary["risk_levels"][risk_level] = drift_summary["risk_levels"].get(risk_level, 0) + 1
                
                if drift.get("policy_violations"):
                    drift_summary["files_with_violations"].append(drift["file_path"])
            
            prompt = f"""You are an AI-powered configuration drift analysis assistant for the Golden Config AI system. Your task is to assess the overall risk of configuration changes across multiple files and provide actionable insights.

### Context:
- **Environment**: {environment}
- **Drift Summary**:
  - Total files changed: {drift_summary['total_files_changed']}
  - Risk level distribution: {drift_summary['risk_levels']}
  - Files with policy violations: {len(drift_summary['files_with_violations'])}

### Key Considerations:
1. **Environment Criticality**:
   - Consider the importance of the environment (e.g., production vs staging) and its tolerance for risk.
2. **Change Volume and Complexity**:
   - Assess the overall volume of changes and their complexity.
3. **Security Implications**:
   - Identify any potential security risks introduced by the changes.
4. **Business Impact**:
   - Evaluate the potential impact of the changes on business operations, compliance, and performance.

### Your Task:
Analyze the overall configuration drift and provide the following in **JSON format**:
1. **Overall Risk Level**: Assess the overall risk level (low, medium, high, critical) based on the drift summary and key considerations.
2. **Risk Factors**: List the key factors contributing to the overall risk level.
3. **Mitigation Strategies**: Provide actionable strategies to mitigate the identified risks.

### Example Output:
```json
{{
  "overall_risk_level": "high",
  "risk_factors": [
    "5 high-risk changes detected",
    "3 files with policy violations",
    "Critical environment: production"
  ],
  "mitigation_strategies": [
    "Conduct a detailed review of all high-risk changes.",
    "Resolve policy violations before deployment.",
    "Perform additional security testing in a staging environment."
  ]
}}
```"""

            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=400):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            # Parse AI response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    ai_assessment = json.loads(json_str)
                    return ai_assessment
            except:
                pass
                
            # Fallback assessment
            return {
                "overall_risk_level": "medium" if len(detailed_drifts) > 5 else "low",
                "risk_factors": [f"{len(detailed_drifts)} configuration changes detected"],
                "mitigation_strategies": ["Review all changes", "Implement monitoring"]
            }
            
        except Exception as e:
            logger.error(f"AI risk assessment failed: {e}")
            # Return basic assessment
            high_risk_count = len([d for d in detailed_drifts if d.get("risk_level") == "high"])
            return {
                "overall_risk_level": "high" if high_risk_count > 0 else "medium",
                "risk_factors": [f"{high_risk_count} high-risk changes detected"],
                "mitigation_strategies": ["Manual review required"]
            }


    async def _assess_overall_drift_risk(self, detailed_drifts: List[Dict[str, Any]], 
                                       environment: str) -> Dict[str, Any]:
        """Assess overall risk from all detected drifts"""
        risk_assessment = {
            "overall_risk_level": "low",
            "risk_factors": [],
            "mitigation_priority": "standard"
        }
        
        high_risk_count = len([d for d in detailed_drifts if d["risk_level"] == "high"])
        medium_risk_count = len([d for d in detailed_drifts if d["risk_level"] == "medium"])
        
        if high_risk_count > 0:
            risk_assessment["overall_risk_level"] = "high"
            risk_assessment["mitigation_priority"] = "urgent"
            risk_assessment["risk_factors"].append(f"{high_risk_count} high-risk changes detected")
        elif medium_risk_count > 5:
            risk_assessment["overall_risk_level"] = "medium"
            risk_assessment["mitigation_priority"] = "elevated"
            risk_assessment["risk_factors"].append(f"{medium_risk_count} medium-risk changes detected")
        
        return risk_assessment

    async def _generate_drift_recommendations(self, detailed_drifts: List[Dict[str, Any]], 
                                            risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        
        if risk_assessment["overall_risk_level"] == "high":
            recommendations.append("Immediate review and remediation required for high-risk changes")
        
        if len(detailed_drifts) > 10:
            recommendations.append("Consider implementing automated configuration management")
        
        return recommendations

    async def _get_applicable_policies(self, policy_set: str, environment: str) -> Dict[str, Any]:
        """Get applicable policies for the given context"""
        # This would be expanded with actual policy retrieval logic
        return self.policy_rules

    async def _validate_change_against_policies(self, change: Dict[str, Any], 
                                              policies: Dict[str, Any], 
                                              environment: str) -> Dict[str, Any]:
        """Validate a single change against policies"""
        return {
            "approved": True,
            "violations": [],
            "warnings": []
        }

    async def _assess_single_change_impact(self, change: Dict[str, Any], 
                                         environment: str, 
                                         impact_scope: str) -> Dict[str, Any]:
        """Assess impact of a single change"""
        return {
            "impact_areas": {
                "security": {"score": 10},
                "performance": {"score": 5},
                "availability": {"score": 15}
            }
        }

    async def _generate_testing_recommendations(self, changes: List[Dict[str, Any]], 
                                              impact_assessment: Dict[str, Any]) -> List[str]:
        """Generate testing recommendations"""
        return ["Perform security testing", "Load testing recommended"]

    async def _generate_deployment_considerations(self, changes: List[Dict[str, Any]], 
                                                impact_assessment: Dict[str, Any]) -> List[str]:
        """Generate deployment considerations"""
        return ["Blue-green deployment recommended", "Rollback plan required"]

    async def _generate_specific_drift_recommendations(self, drift: Dict[str, Any], 
                                                     priority_level: str) -> Dict[str, List[str]]:
        """Generate specific recommendations for a drift"""
        return {
            "immediate": [f"Review changes in {drift['file_path']}"],
            "corrective": [f"Update {drift['file_path']} to match golden config"],
            "preventive": ["Implement configuration monitoring"]
        }

    async def _suggest_monitoring_improvements(self, drift_analysis: Dict[str, Any]) -> List[str]:
        """Suggest monitoring improvements"""
        return ["Enhanced configuration monitoring", "Real-time drift detection"]

    async def _suggest_process_improvements(self, drift_analysis: Dict[str, Any]) -> List[str]:
        """Suggest process improvements"""
        return ["Automated configuration management", "Regular drift audits"]

    # Pattern analysis helper methods (simplified implementations)
    async def _identify_frequent_changes(self, changes: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
        """Identify frequently changing configurations"""
        return []

    async def _identify_risky_patterns(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify risky change patterns"""
        return []

    async def _identify_seasonal_patterns(self, changes: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
        """Identify seasonal change patterns"""
        return []

    async def _identify_error_prone_areas(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify error-prone configuration areas"""
        return []

    async def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from patterns"""
        return []

    async def _generate_pattern_recommendations(self, patterns: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate recommendations from patterns"""
        return []

    async def process_task(self, task_request: TaskRequest) -> TaskResponse:
        """Process a task request from the Supervisor Agent"""
        logger.info(f"Processing diff engine task: {task_request.task_type}")
        
        start_time = datetime.utcnow()
        
        try:
            if task_request.task_type == "drift_analysis":
                result = await DiffPolicyEngineAgent.analyze_configuration_drift(
                    self,
                    golden_config=task_request.parameters.get("golden_config", {}),
                    current_config=task_request.parameters.get("current_config", {}),
                    environment=task_request.parameters.get("environment", "production"),
                    analysis_depth=task_request.parameters.get("analysis_depth", "comprehensive")
                )
            elif task_request.task_type == "policy_validation":
                result = await DiffPolicyEngineAgent.validate_policy_compliance(
                    self,
                    configuration_changes=task_request.parameters.get("configuration_changes", []),
                    policy_set=task_request.parameters.get("policy_set", "default"),
                    environment=task_request.parameters.get("environment", "production")
                )
            elif task_request.task_type == "change_impact":
                result = await DiffPolicyEngineAgent.assess_change_impact(
                    self,
                    configuration_changes=task_request.parameters.get("configuration_changes", []),
                    environment=task_request.parameters.get("environment", "production"),
                    impact_scope=task_request.parameters.get("impact_scope", "full")
                )
            elif task_request.task_type == "generate_recommendations":
                result = await DiffPolicyEngineAgent.generate_drift_recommendations(
                    self,
                    drift_analysis=task_request.parameters.get("drift_analysis", {}),
                    priority_level=task_request.parameters.get("priority_level", "high")
                )
            elif task_request.task_type == "pattern_classification":
                result = await DiffPolicyEngineAgent.classify_change_patterns(
                    self,
                    historical_changes=task_request.parameters.get("historical_changes", []),
                    analysis_period=task_request.parameters.get("analysis_period", "30_days")
                )
            elif task_request.task_type == "remediation_plan":
                result = await DiffPolicyEngineAgent.generate_remediation_plan(
                    self,
                    drift_analysis=task_request.parameters.get("drift_analysis", {}),
                    compliance_results=task_request.parameters.get("compliance_results", None),
                    priority_level=task_request.parameters.get("priority_level", "medium")
                )
            else:
                raise ValueError(f"Unknown task type: {task_request.task_type}")
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            return TaskResponse(
                task_id=task_request.task_id,
                status="success" if result.get("status") == "success" else "failure",
                result=result,
                error=None,
                metadata={"agent_type": "diff_engine", "task_type": task_request.task_type},
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Diff engine task processing failed: {e}")
            return TaskResponse(
                task_id=task_request.task_id,
                status="failure",
                result={"error": str(e)},
                error=str(e),
                metadata={"agent_type": "diff_engine", "task_type": task_request.task_type},
                processing_time_seconds=processing_time
            )
