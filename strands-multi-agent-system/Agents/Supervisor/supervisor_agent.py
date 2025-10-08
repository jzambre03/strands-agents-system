"""
Supervisor Agent
Orchestrates the validation workflow and coordinates worker agents.

This agent:
1. Receives validation requests (project_id, mr_iid)
2. Creates validation run
3. Orchestrates worker agents via Strands Graph
4. Aggregates results from all agents
5. Formats comprehensive MR comment
6. Makes final business decision
7. Maintains audit trail
"""

from datetime import datetime
from typing import Dict, Any
import logging
import json
import glob

from strands import Agent, tool
from strands.models import BedrockModel

from shared.config import Config

logger = logging.getLogger(__name__)

# Initialize config
config = Config()


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are the Supervisor Agent - the orchestration brain for Golden Config validation.

Your core responsibilities:
1. Coordinate the validation workflow across 2 specialized worker agents
2. Create validation runs with unique identifiers
3. Execute the multi-agent workflow (Config Collector → Diff Engine)
4. Aggregate results from all agents
5. Make final business decisions (PASS/FAIL)
6. Format comprehensive, actionable reports
7. Maintain audit trail of all decisions

Your expertise:
- Workflow orchestration and coordination
- Multi-agent system management
- Business rule application for PASS/FAIL decisions
- Clear communication and reporting
- Error handling and recovery

Your workflow:
1. Start by creating a unique validation run ID
2. Execute the 2-agent pipeline:
   - Config Collector: Fetches and parses configuration diffs
   - Diff Engine: Analyzes drift and validates against policies
3. Load and review outputs from both agents
4. Aggregate policy violations and drift findings
5. Apply business logic for final approval decision
6. Format a clear, professional report with:
   - Verdict (PASS/FAIL)
   - Summary of findings
   - Detailed violations (grouped by severity)
   - Remediation suggestions
   - Run metadata
7. Return comprehensive validation result

You are the final authority. Be thorough, fair, and communicative.
Always maintain the complete audit trail.
"""


# ============================================================================
# AGENT-SPECIFIC TOOLS
# ============================================================================

@tool
def create_validation_run(
    project_id: str,
    mr_iid: str,
    source_branch: str,
    target_branch: str
) -> dict:
    """
    Create a new validation run.
    
    Args:
        project_id: Project identifier
        mr_iid: Merge request or validation ID
        source_branch: Source branch name
        target_branch: Target branch name
        
    Returns:
        Unique run ID for this validation
        
    Example:
        >>> create_validation_run("myorg/myrepo", "123", "feature", "gold")
    """
    try:
        # Generate unique run ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"run_{timestamp}_{mr_iid}"
        
        logger.info(f"Created validation run: {run_id}")
        
        return {
            "success": True,
            "message": f"Created validation run: {run_id}",
            "data": {"run_id": run_id}
        }
        
    except Exception as e:
        logger.error(f"Failed to create validation run: {e}")
        return {
            "success": False,
            "error": f"Failed to create validation run: {str(e)}"
        }


@tool
def execute_worker_pipeline(
    project_id: str,
    mr_iid: str,
    run_id: str,
    repo_url: str,
    golden_branch: str,
    drift_branch: str,
    target_folder: str = ""
) -> dict:
    """
    Execute the 2-agent file-based pipeline with policy-aware analysis.
    
    This orchestrates two worker agents in sequence using file-based communication:
    1. Config Collector - Fetches diffs from Git, saves to context_bundle.json (with drift.py precision)
    2. Diff Engine - Reads context_bundle, analyzes with policy-aware AI, saves to enhanced_analysis.json
    
    Args:
        project_id: Project identifier
        mr_iid: Merge request or validation ID
        run_id: Validation run ID
        repo_url: GitLab repository URL
        golden_branch: Golden/reference branch name
        drift_branch: Drift/comparison branch name
        target_folder: Optional subfolder to analyze
        
    Returns:
        Workflow execution result with file paths
        
    Example:
        >>> execute_worker_pipeline("myorg/myrepo", "123", "run_xyz", 
        ...     "https://gitlab.verizon.com/saja9l7/golden_config.git", "gold", "drift")
    """
    try:
        logger.info(f"Starting 2-agent file-based pipeline for run {run_id}")
        
        # Import agents
        from Agents.workers.config_collector.config_collector_agent import ConfigCollectorAgent
        from Agents.workers.diff_policy_engine.diff_engine_agent import DiffPolicyEngineAgent
        from shared.models import TaskRequest
        
        results = {}
        
        # Step 1: Config Collector (saves to file)
        logger.info("Step 1: Running Config Collector Agent")
        collector = ConfigCollectorAgent(config)
        
        collector_task = TaskRequest(
            task_id=f"{run_id}_collector",
            task_type="collect_diffs",
            parameters={
                "repo_url": repo_url,
                "golden_branch": golden_branch,
                "drift_branch": drift_branch,
                "target_folder": target_folder
            }
        )
        
        collector_result = collector.process_task(collector_task)
        
        if collector_result.status != "success":
            logger.error(f"Config Collector failed: {collector_result.error}")
            return {
                "success": False,
                "error": f"Config Collector failed: {collector_result.error}",
                "data": {
                    "status": "failed",
                    "completed_agents": 0,
                    "failed_agents": 1,
                    "failed_at": "config_collector"
                }
            }
        
        # Get context bundle file path (NEW: Phase 4)
        context_bundle_file = collector_result.result.get("context_bundle_file")
        logger.info(f"✅ Config Collector completed: {context_bundle_file}")
        
        results['collector'] = {
            "status": "success",
            "context_bundle_file": context_bundle_file,  # NEW: context_bundle instead of drift_analysis
            "summary": collector_result.result.get("summary", {})
        }
        
        # Step 2: Diff Engine (reads context_bundle, analyzes with policy, saves to enhanced_analysis)
        logger.info("Step 2: Running Diff Engine Agent (policy-aware)")
        diff_engine = DiffPolicyEngineAgent(config)
        
        diff_task = TaskRequest(
            task_id=f"{run_id}_diff_engine",
            task_type="analyze_drift",
            parameters={
                "context_bundle_file": context_bundle_file  # NEW: pass context_bundle
            }
        )
        
        diff_result = diff_engine.process_task(diff_task)
        
        if diff_result.status != "success":
            logger.error(f"Diff Engine failed: {diff_result.error}")
            return {
                "success": False,
                "error": f"Diff Engine failed: {diff_result.error}",
                "data": {
                    "status": "partial",
                    "completed_agents": 1,
                    "failed_agents": 1,
                    "failed_at": "diff_engine",
                    "results": results
                }
            }
        
        # Get enhanced analysis file path (NEW: Phase 4)
        enhanced_analysis_file = diff_result.result.get("enhanced_analysis_file")
        logger.info(f"✅ Diff Engine completed: {enhanced_analysis_file}")
        
        results['diff_engine'] = {
            "status": "success",
            "enhanced_analysis_file": enhanced_analysis_file,  # NEW: enhanced_analysis
            "summary": diff_result.result.get("summary", {})
        }
        
        logger.info(f"✅ Pipeline completed successfully (2/2 agents)")
        
        return {
            "success": True,
            "message": "Pipeline completed: 2/2 agents succeeded",
            "data": {
                "status": "completed",
                "completed_agents": 2,
                "failed_agents": 0,
                "context_bundle_file": context_bundle_file,        # NEW
                "enhanced_analysis_file": enhanced_analysis_file,  # NEW
                "results": results
            }
        }
        
    except Exception as e:
        logger.exception(f"Worker pipeline execution failed: {e}")
        return {
            "success": False,
            "error": f"Pipeline execution failed: {str(e)}"
        }


@tool
def aggregate_validation_results(
    collector_results: dict,
    diff_engine_results: dict
) -> dict:
    """
    Aggregate results from the 2 worker agents (with policy-aware analysis).
    
    Args:
        collector_results: Results from Config Collector Agent (contains context_bundle_file)
        diff_engine_results: Results from Diff Engine Agent (contains enhanced_analysis_file)
        
    Returns:
        Aggregated results from both agents with intelligent verdict and policy violations
        
    Example:
        >>> aggregate_validation_results({...}, {...})
    """
    try:
        import json
        from pathlib import Path
        
        logger.info("Aggregating results from 2 agents (policy-aware)")
        
        # Get file paths from agent results (NEW: Phase 4)
        context_bundle_file = collector_results.get("context_bundle_file")
        enhanced_analysis_file = diff_engine_results.get("enhanced_analysis_file")
        
        if not context_bundle_file or not enhanced_analysis_file:
            logger.error("Missing analysis files from agents")
            return {
                "success": False,
                "error": "Missing context_bundle_file or enhanced_analysis_file from agent results"
            }
        
        # Load context bundle (from Config Collector - NEW: Phase 4)
        logger.info(f"Loading context bundle from: {context_bundle_file}")
        try:
            with open(context_bundle_file, 'r', encoding='utf-8') as f:
                context_bundle = json.load(f)
            
            # Extract data from context_bundle structure
            overview = context_bundle.get("overview", {})
            deltas = context_bundle.get("deltas", [])
            file_changes = context_bundle.get("file_changes", {})
            
            files_with_drift = len(file_changes.get("modified", [])) + len(file_changes.get("added", [])) + len(file_changes.get("removed", []))
            total_files_compared = overview.get("files_compared", 0)
            environment = overview.get("environment", "production")
            
        except Exception as e:
            logger.error(f"Failed to load context bundle file: {e}")
            return {
                "success": False,
                "error": f"Failed to load context bundle file: {str(e)}"
            }
        
        # Load enhanced analysis (from Diff Engine - NEW: Phase 4)
        logger.info(f"Loading enhanced analysis from: {enhanced_analysis_file}")
        try:
            with open(enhanced_analysis_file, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
            
            ai_policy_analysis = enhanced_data.get("ai_policy_analysis", {})
            analyzed_deltas_with_ai = enhanced_data.get("analyzed_deltas_with_ai", [])
            clusters = enhanced_data.get("clusters", [])  # NEW: Clustered deltas
            
        except Exception as e:
            logger.error(f"Failed to load enhanced analysis file: {e}")
            return {
                "success": False,
                "error": f"Failed to load enhanced analysis file: {str(e)}"
            }
        
        # Extract key metrics from enhanced analysis
        policy_violations = ai_policy_analysis.get("policy_violations", [])
        overall_risk_level = ai_policy_analysis.get("overall_risk_level", "unknown")
        risk_assessment = ai_policy_analysis.get("risk_assessment", {})
        recommendations = ai_policy_analysis.get("recommendations", [])
        
        # Intelligent verdict logic
        verdict = determine_verdict(
            files_with_drift=files_with_drift,
            overall_risk_level=overall_risk_level,
            policy_violations=policy_violations,
            environment=environment
        )
        
        # Count violations by severity
        critical_violations = len([v for v in policy_violations if v.get('severity') == 'critical'])
        high_violations = len([v for v in policy_violations if v.get('severity') == 'high'])
        
        # Aggregate results (NEW: Phase 4 format)
        aggregated = {
            "files_analyzed": total_files_compared,
            "files_compared": total_files_compared,
            "files_with_drift": files_with_drift,
            "total_deltas": len(deltas),
            "deltas_analyzed": len(analyzed_deltas_with_ai),
            "total_clusters": len(clusters),  # NEW: Clustering feature
            "policy_violations_count": len(policy_violations),
            "policy_violations": policy_violations,
            "critical_violations": critical_violations,
            "high_violations": high_violations,
            "overall_risk_level": overall_risk_level,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "verdict": verdict,
            "environment": environment,
            "clusters": clusters,  # NEW: Clustered deltas with root causes
            "analyzed_deltas": analyzed_deltas_with_ai,  # NEW: structured deltas with AI analysis
            "deltas_with_patches": [d for d in analyzed_deltas_with_ai if d.get('patch_hint')],  # NEW: deltas with patch hints
            "file_paths": {
                "context_bundle": context_bundle_file,      # NEW
                "enhanced_analysis": enhanced_analysis_file  # NEW
            }
        }
        
        logger.info(f"✅ Aggregated results: {verdict} (Risk: {overall_risk_level}, Violations: {len(policy_violations)})")
        
        # Save full aggregated results to file to avoid max_tokens issues
        from pathlib import Path
        aggregated_dir = Path("config_data/aggregated_results")
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        
        aggregated_file = aggregated_dir / f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)
        
        logger.info(f"Saved full aggregated results to: {aggregated_file}")
        
        # Return only minimal summary to avoid max_tokens when passing to format_validation_report
        return {
            "success": True,
            "message": f"Aggregated results: {verdict}",
            "data": {
                "verdict": verdict,
                "files_with_drift": files_with_drift,
                "overall_risk_level": overall_risk_level,
                "policy_violations_count": len(policy_violations),
                "critical_violations": critical_violations,
                "high_violations": high_violations,
                "aggregated_file": str(aggregated_file),  # Pass file path instead of full data
                # Essential data for report generation
                "file_paths": aggregated["file_paths"]
            }
        }
        
    except Exception as e:
        logger.exception(f"Failed to aggregate results: {e}")
        return {
            "success": False,
            "error": f"Failed to aggregate results: {str(e)}"
        }


def determine_verdict(
    files_with_drift: int,
    overall_risk_level: str,
    policy_violations: list,
    environment: str
) -> str:
    """
    Determine intelligent verdict based on drift, risk, violations, and environment.
    
    Business Logic:
    - PASS: No drift detected
    - BLOCK: Critical violations or high risk in production
    - REVIEW_REQUIRED: Medium risk or violations that need human review
    - WARN: Low risk changes that should be reviewed but not blocking
    
    Args:
        files_with_drift: Number of files with configuration drift
        overall_risk_level: AI-assessed overall risk (low, medium, high, critical)
        policy_violations: List of policy violations
        environment: Target environment (production, staging, development)
        
    Returns:
        Verdict string (PASS, BLOCK, REVIEW_REQUIRED, WARN)
    """
    # No drift = always pass
    if files_with_drift == 0:
        return "PASS"
    
    # Check for critical violations
    critical_violations = [v for v in policy_violations if isinstance(v, dict) and v.get('severity') == 'critical']
    if critical_violations:
        return "BLOCK"
    
    # Environment-specific rules
    if environment == "production":
        # Production has lower risk tolerance
        if overall_risk_level == "critical":
            return "BLOCK"
        elif overall_risk_level == "high":
            return "BLOCK"
        elif overall_risk_level == "medium":
            # Medium risk in production requires review
            high_violations = [v for v in policy_violations if v.get('severity') == 'high']
            if high_violations:
                return "REVIEW_REQUIRED"
            else:
                return "WARN"
        elif overall_risk_level == "low":
            if policy_violations:
                return "WARN"
            else:
                return "WARN"  # Even low-risk changes should be reviewed in production
    
    elif environment in ["staging", "pre-production"]:
        # Staging can tolerate more risk
        if overall_risk_level == "critical":
            return "BLOCK"
        elif overall_risk_level == "high":
            return "REVIEW_REQUIRED"
        elif overall_risk_level == "medium":
            return "WARN"
        else:
            return "WARN"
    
    else:  # development, testing, etc.
        # Development environments are more permissive
        if overall_risk_level == "critical":
            return "BLOCK"
        elif overall_risk_level == "high":
            return "REVIEW_REQUIRED"
        else:
            return "WARN"
    
    # Default fallback
    return "REVIEW_REQUIRED"


@tool
def format_validation_report(
    run_id: str,
    aggregated_results: dict
) -> dict:
    """
    Format comprehensive validation report with detailed violations and recommendations.
    
    Args:
        run_id: Validation run ID
        aggregated_results: Aggregated results from aggregate_validation_results
        
    Returns:
        Formatted markdown report with full details
        
    Example:
        >>> format_validation_report("run_xyz", aggregated_results)
    """
    try:
        # Load full aggregated data from file if available
        aggregated_file = aggregated_results.get("aggregated_file")
        if aggregated_file:
            import json
            from pathlib import Path
            with open(aggregated_file, 'r') as f:
                full_aggregated = json.load(f)
            logger.info(f"Loaded full aggregated results from: {aggregated_file}")
        else:
            # Fallback to passed data (for backward compatibility)
            full_aggregated = aggregated_results
        
        # Extract data from aggregated results
        verdict = aggregated_results.get("verdict", "UNKNOWN")
        files_with_drift = aggregated_results.get("files_with_drift", 0)
        policy_violations = full_aggregated.get("policy_violations", [])
        critical_violations = aggregated_results.get("critical_violations", 0)
        high_violations = aggregated_results.get("high_violations", 0)
        risk_level = aggregated_results.get("overall_risk_level", "unknown")
        risk_assessment = full_aggregated.get("risk_assessment", {})
        recommendations = full_aggregated.get("recommendations", [])
        environment = full_aggregated.get("environment", "unknown")
        file_paths = aggregated_results.get("file_paths", {})
        
        # Verdict emoji and color
        verdict_emoji = {
            "PASS": "✅",
            "WARN": "⚠️",
            "REVIEW_REQUIRED": "🔍",
            "BLOCK": "🚫"
        }.get(verdict, "❓")
        
        # Risk emoji
        risk_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }.get(risk_level, "⚪")
        
        # Build summary
        if verdict == "PASS":
            summary = "✅ **Configuration validated successfully.** No drift detected."
        elif verdict == "BLOCK":
            summary = f"🚫 **BLOCKED:** Critical issues detected. Deployment must be prevented."
        elif verdict == "REVIEW_REQUIRED":
            summary = f"🔍 **REVIEW REQUIRED:** Changes need human approval before deployment."
        elif verdict == "WARN":
            summary = f"⚠️ **WARNING:** Low-risk changes detected. Review recommended."
        else:
            summary = f"Found {files_with_drift} file(s) with drift"
        
        # Format report header
        report = f"""
# {verdict_emoji} Configuration Validation Report

**Run ID:** `{run_id}`  
**Verdict:** **{verdict}**  
**Environment:** **{environment}**  
**Risk Level:** {risk_emoji} **{risk_level.upper()}**  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Executive Summary

{summary}

### Key Metrics
- **Files Analyzed:** {full_aggregated.get('files_analyzed', 0)}
- **Files with Drift:** {files_with_drift}
- **Total Deltas Detected:** {full_aggregated.get('total_deltas', 0)}
- **Deltas Analyzed (AI):** {full_aggregated.get('deltas_analyzed', 0)}
- **Clusters Identified:** {full_aggregated.get('total_clusters', 0)}
- **Pinpoint Locations:** {len([d for d in full_aggregated.get('analyzed_deltas', []) if d.get('pinpoint')])}
- **Evidence Checks:** {len([d for d in full_aggregated.get('analyzed_deltas', []) if d.get('evidence_check')])}
- **Policy Violations:** {len(policy_violations)} ({critical_violations} critical, {high_violations} high)
- **Overall Risk:** {risk_level}
- **Analysis Type:** Policy-Aware (drift.py + AI + Clustering + Pinpoint + Evidence)

---
"""
        
        # Add clusters section if any (NEW: Clustering feature)
        clusters = full_aggregated.get('clusters', [])
        if clusters:
            report += "\n## 🔗 Change Clusters (Grouped by Root Cause)\n\n"
            report += "*Related changes grouped together for better insights:*\n\n"
            
            for i, cluster in enumerate(clusters, 1):
                cluster_id = cluster.get('id', 'unknown')
                root_cause = cluster.get('root_cause', 'Unknown cause')
                items = cluster.get('items', [])
                severity = cluster.get('severity', 'medium')
                verdict = cluster.get('verdict', 'DRIFT_WARN')
                cluster_type = cluster.get('type', 'unknown')
                confidence = cluster.get('confidence', 0.0)
                
                # Emoji based on severity
                severity_emoji = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(severity, '⚪')
                
                verdict_emoji = {
                    'DRIFT_BLOCKING': '🚫',
                    'DRIFT_WARN': '⚠️',
                    'NEW_BUILD_OK': '✅',
                    'NO_DRIFT': '✅'
                }.get(verdict, '❓')
                
                report += f"### {i}. {severity_emoji} {root_cause}\n\n"
                report += f"- **Cluster ID:** `{cluster_id}`\n"
                report += f"- **Type:** {cluster_type.replace('_', ' ').title()}\n"
                report += f"- **Severity:** {severity.upper()}\n"
                report += f"- **Verdict:** {verdict_emoji} {verdict}\n"
                report += f"- **Confidence:** {confidence:.0%}\n"
                report += f"- **Affected Items:** {len(items)} deltas\n"
                
                # Show affected files if available
                if 'files' in cluster:
                    files = cluster['files']
                    report += f"- **Affected Files:** {', '.join(files[:5])}"
                    if len(files) > 5:
                        report += f" (and {len(files) - 5} more)"
                    report += "\n"
                elif 'file' in cluster:
                    report += f"- **File:** {cluster['file']}\n"
                
                # Show pattern if available
                if 'pattern' in cluster:
                    report += f"- **Pattern:** {cluster['pattern'].replace('_', ' ').title()}\n"
                
                # Show ecosystem if available
                if 'ecosystem' in cluster:
                    report += f"- **Ecosystem:** {cluster['ecosystem']}\n"
                
                report += f"\n**Related Changes:**\n"
                for item_id in items[:10]:  # Show first 10 items
                    report += f"- `{item_id}`\n"
                if len(items) > 10:
                    report += f"- *(and {len(items) - 10} more...)*\n"
                
                report += "\n"
            
            report += "---\n\n"
        
        # Add policy violations section if any
        if policy_violations:
            report += "\n## 🚨 Policy Violations\n\n"
            
            # Group violations by severity
            violations_by_severity = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            for violation in policy_violations:
                severity = violation.get('severity', 'medium').lower()
                if severity in violations_by_severity:
                    violations_by_severity[severity].append(violation)
            
            # Critical violations
            if violations_by_severity['critical']:
                report += "### 🔴 Critical Violations\n\n"
                for i, v in enumerate(violations_by_severity['critical'], 1):
                    report += f"{i}. **{v.get('type', 'Unknown').upper()}:** {v.get('description', 'No description')}\n"
                    if v.get('rule'):
                        report += f"   - **Rule:** `{v['rule']}`\n"
                report += "\n"
            
            # High severity violations
            if violations_by_severity['high']:
                report += "### 🟠 High Severity Violations\n\n"
                for i, v in enumerate(violations_by_severity['high'], 1):
                    report += f"{i}. **{v.get('type', 'Unknown').upper()}:** {v.get('description', 'No description')}\n"
                    if v.get('rule'):
                        report += f"   - **Rule:** `{v['rule']}`\n"
                report += "\n"
            
            # Medium/Low violations
            other_violations = violations_by_severity['medium'] + violations_by_severity['low']
            if other_violations:
                report += "### 🟡 Other Violations\n\n"
                for i, v in enumerate(other_violations, 1):
                    report += f"{i}. **{v.get('type', 'Unknown')}:** {v.get('description', 'No description')}\n"
                report += "\n"
            
            report += "---\n\n"
        
        # Add patch hints section (NEW: Copy-pasteable fixes)
        deltas_with_patches = aggregated_results.get('deltas_with_patches', [])
        if deltas_with_patches:
            report += "## 🔧 Patch Hints (Copy-Pasteable Fixes)\n\n"
            report += "*Copy these snippets directly into your files to fix the issues:*\n\n"
            
            for i, delta in enumerate(deltas_with_patches, 1):
                patch = delta.get('patch_hint', {})
                if not patch:
                    continue
                
                file = delta.get('file', 'unknown')
                verdict = delta.get('verdict', 'unknown')
                patch_type = patch.get('type', 'generic')
                patch_content = patch.get('content', '')
                
                # Only show patches for blocked or warn items
                if verdict in ['DRIFT_BLOCKING', 'DRIFT_WARN']:
                    report += f"### {i}. {file}\n\n"
                    report += f"**Issue:** {verdict}\n\n"
                    
                    # Format based on patch type
                    if patch_type == 'yaml_snippet':
                        report += "```yaml\n"
                    elif patch_type == 'json_snippet':
                        report += "```json\n"
                    elif patch_type == 'unified_diff':
                        report += "```diff\n"
                    elif patch_type == 'properties_snippet':
                        report += "```properties\n"
                    elif patch_type == 'dependency_update':
                        report += "```\n"
                    else:
                        report += "```\n"
                    
                    report += f"{patch_content}\n"
                    report += "```\n\n"
            
            report += "---\n\n"
        
        # Add pinpoint locations section (NEW: Feature #3)
        deltas_with_pinpoints = [d for d in aggregated_results.get('analyzed_deltas', []) if d.get('pinpoint')]
        if deltas_with_pinpoints:
            report += "## 📍 Pinpoint Locations (Quick Navigation)\n\n"
            report += "*Click these links to jump directly to the issues in your IDE:*\n\n"
            
            for i, delta in enumerate(deltas_with_pinpoints[:10], 1):  # Show top 10
                pinpoint = delta.get('pinpoint', {})
                file = pinpoint.get('file', 'unknown')
                location_string = pinpoint.get('location_string', file)
                human_readable = pinpoint.get('human_readable', f"{file}")
                navigation = pinpoint.get('navigation', {})
                nav_type = navigation.get('type', 'Location')
                search_hint = navigation.get('search_hint', f'Search in {file}')
                ide_links = pinpoint.get('ide_links', {})
                
                report += f"### {i}. {human_readable}\n\n"
                report += f"- **Location:** `{location_string}`\n"
                report += f"- **Type:** {nav_type}\n"
                report += f"- **Quick Access:** {search_hint}\n"
                
                # Add IDE links if available
                if ide_links:
                    report += "- **IDE Links:**\n"
                    for ide, link in ide_links.items():
                        ide_name = ide.replace('_', ' ').title()
                        report += f"  - [{ide_name}]({link})\n"
                
                # Add VS Code command
                vs_code_cmd = navigation.get('vs_code_command', '')
                if vs_code_cmd:
                    report += f"- **VS Code:** {vs_code_cmd}\n"
                
                # Add Vim command
                vim_cmd = navigation.get('vim_command', '')
                if vim_cmd:
                    report += f"- **Vim:** {vim_cmd}\n"
                
                report += "\n"
            
            if len(deltas_with_pinpoints) > 10:
                report += f"*... and {len(deltas_with_pinpoints) - 10} more locations*\n\n"
            
            report += "---\n\n"
        
        # Add evidence checking section (NEW: Feature #4)
        deltas_with_evidence = [d for d in aggregated_results.get('analyzed_deltas', []) if d.get('evidence_check')]
        if deltas_with_evidence:
            report += "## 🔍 Evidence Checking (Approval Requirements)\n\n"
            report += "*Validation against required approvals and evidence:*\n\n"
            
            # Group by approval status
            approved = [d for d in deltas_with_evidence if d.get('evidence_check', {}).get('approval_status') == 'approved']
            partial = [d for d in deltas_with_evidence if d.get('evidence_check', {}).get('approval_status') == 'partial_approval']
            pending = [d for d in deltas_with_evidence if d.get('evidence_check', {}).get('approval_status') == 'pending_review']
            rejected = [d for d in deltas_with_evidence if d.get('evidence_check', {}).get('approval_status') == 'rejected']
            
            if rejected:
                report += f"### ❌ Blocked Changes ({len(rejected)})\n\n"
                for i, delta in enumerate(rejected[:5], 1):
                    evidence_check = delta.get('evidence_check', {})
                    file = evidence_check.get('file', 'unknown')
                    location = evidence_check.get('location', '')
                    compliance_score = evidence_check.get('compliance_score', 0.0)
                    summary = evidence_check.get('validation_summary', '')
                    
                    report += f"#### {i}. {file}\n\n"
                    report += f"- **Location:** `{location}`\n"
                    report += f"- **Compliance:** {compliance_score:.0%}\n"
                    report += f"- **Status:** {summary}\n"
                    
                    # Show missing evidence
                    missing = evidence_check.get('evidence_missing', [])
                    if missing:
                        report += f"- **Missing Evidence:**\n"
                        for missing_req in missing[:3]:  # Show first 3
                            req_type = missing_req.get('requirement', '')
                            description = missing_req.get('description', '')
                            priority = missing_req.get('priority', '')
                            ticket_type = missing_req.get('ticket_type', '')
                            
                            priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
                            report += f"  - {priority_emoji} {description}\n"
                            report += f"    Required: {ticket_type}\n"
                    
                    report += "\n"
                
                if len(rejected) > 5:
                    report += f"*... and {len(rejected) - 5} more blocked changes*\n\n"
                
                report += "---\n\n"
            
            if pending:
                report += f"### 🔄 Pending Review ({len(pending)})\n\n"
                for i, delta in enumerate(pending[:3], 1):
                    evidence_check = delta.get('evidence_check', {})
                    file = evidence_check.get('file', 'unknown')
                    location = evidence_check.get('location', '')
                    summary = evidence_check.get('validation_summary', '')
                    
                    report += f"#### {i}. {file}\n\n"
                    report += f"- **Location:** `{location}`\n"
                    report += f"- **Status:** {summary}\n"
                    
                    # Show found evidence
                    found = evidence_check.get('evidence_found', [])
                    if found:
                        report += f"- **Evidence Found:**\n"
                        for evidence in found[:2]:  # Show first 2
                            evidence_id = evidence.get('evidence_id', '')
                            evidence_type = evidence.get('evidence_type', '')
                            description = evidence.get('description', '')
                            report += f"  - ✅ {evidence_type}: {description}\n"
                    
                    report += "\n"
                
                report += "---\n\n"
            
            if partial:
                report += f"### ⚠️ Partial Approval ({len(partial)})\n\n"
                for i, delta in enumerate(partial[:3], 1):
                    evidence_check = delta.get('evidence_check', {})
                    file = evidence_check.get('file', 'unknown')
                    location = evidence_check.get('location', '')
                    summary = evidence_check.get('validation_summary', '')
                    
                    report += f"#### {i}. {file}\n\n"
                    report += f"- **Location:** `{location}`\n"
                    report += f"- **Status:** {summary}\n\n"
                
                report += "---\n\n"
            
            if approved:
                report += f"### ✅ Fully Approved ({len(approved)})\n\n"
                for i, delta in enumerate(approved[:3], 1):
                    evidence_check = delta.get('evidence_check', {})
                    file = evidence_check.get('file', 'unknown')
                    location = evidence_check.get('location', '')
                    summary = evidence_check.get('validation_summary', '')
                    
                    report += f"#### {i}. {file}\n\n"
                    report += f"- **Location:** `{location}`\n"
                    report += f"- **Status:** {summary}\n\n"
                
                report += "---\n\n"
        
        # Add risk assessment section
        if risk_assessment:
            report += "## 📊 Risk Assessment\n\n"
            
            # Safety check: ensure risk_assessment is a dict
            if isinstance(risk_assessment, str):
                report += f"{risk_assessment}\n\n"
            elif isinstance(risk_assessment, dict):
                risk_factors = risk_assessment.get('risk_factors', [])
                if risk_factors:
                    report += "### Risk Factors\n\n"
                    for factor in risk_factors:
                        report += f"- {factor}\n"
                    report += "\n"
                
                mitigation_strategies = risk_assessment.get('mitigation_strategies', [])
                if mitigation_strategies:
                    priority = risk_assessment.get('mitigation_priority', 'standard')
                    priority_emoji = "🚨" if priority == "urgent" else "📌"
                    
                    report += f"### {priority_emoji} Mitigation Strategies (Priority: {priority.upper()})\n\n"
                    for i, strategy in enumerate(mitigation_strategies, 1):
                        report += f"{i}. {strategy}\n"
                    report += "\n"
            
            report += "---\n\n"
        
        # Add recommendations section
        if recommendations:
            report += "## 🔧 Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                if isinstance(rec, dict):
                    priority = rec.get('priority', 'medium')
                    action = rec.get('action', str(rec))
                    rationale = rec.get('rationale', '')
                    report += f"{i}. **[{priority.upper()}]** {action}\n"
                    if rationale:
                        report += f"   - *{rationale}*\n"
                else:
                    report += f"{i}. {rec}\n"
            report += "\n---\n\n"
        
        # Add next steps based on verdict
        report += "## 🎯 Next Steps\n\n"
        if verdict == "PASS":
            report += "✅ No action required. Configuration is compliant.\n\n"
        elif verdict == "BLOCK":
            report += """
🚫 **DEPLOYMENT BLOCKED**

**Required Actions:**
1. Review all critical violations listed above
2. Revert changes or fix violations
3. Re-run validation before attempting deployment
4. Do not proceed to production with these issues

**If you believe this is a false positive, contact the security team.**
"""
        elif verdict == "REVIEW_REQUIRED":
            report += """
🔍 **MANUAL REVIEW REQUIRED**

**Required Actions:**
1. Review the policy violations and risk assessment
2. Determine if changes are acceptable for the target environment
3. If approved, document the decision and proceed
4. If rejected, revert changes and re-validate

**Approval from team lead or security team may be required.**
"""
        elif verdict == "WARN":
            report += """
⚠️ **REVIEW RECOMMENDED**

**Suggested Actions:**
1. Review the changes to ensure they are intentional
2. Verify changes in staging environment before production
3. Monitor for any unexpected behavior after deployment
4. Consider adding these changes to your golden config

**Deployment may proceed with appropriate review.**
"""
        
        report += "\n---\n\n"
        
        # Add file references (NEW: Phase 4)
        report += "## 📁 Analysis Files\n\n"
        if file_paths:
            report += f"- **Context Bundle:** `{file_paths.get('context_bundle', 'N/A')}`\n"
            report += f"- **Enhanced Analysis:** `{file_paths.get('enhanced_analysis', 'N/A')}`\n"
        report += "\n"
        
        # Add pipeline info
        report += """## 🔄 Validation Pipeline

```
Supervisor Agent (Claude 3.5 Sonnet)
    ↓
    ├─► Config Collector Agent (Claude 3 Haiku)
    │   └─► drift.py precision analysis → context_bundle.json
    │       • Line-precise locators (yamlpath, jsonpath)
    │       • Structured deltas
    │       • Dependency analysis
    │
    └─► Diff Policy Engine Agent (Claude 3 Haiku)
        └─► Policy-aware AI analysis → enhanced_analysis.json
            • Policy tag evaluation (invariant_breach, allowed_variance, suspect)
            • Verdict generation (NO_DRIFT, NEW_BUILD_OK, DRIFT_WARN, DRIFT_BLOCKING)
            • Confidence scores
```

*Powered by AWS Strands Multi-Agent Framework + drift.py Precision Analysis*

---

**Report Generated:** {timestamp}
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        logger.info(f"✅ Formatted comprehensive validation report for run {run_id}")
        
        # Auto-save the report to avoid passing large data through AI tools
        from pathlib import Path
        report_dir = Path("config_data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"{run_id}_report.md"
        report_file.write_text(report)
        logger.info(f"Auto-saved report to {report_file}")
        
        # Return minimal response to avoid max_tokens issues
        return {
            "success": True,
            "message": f"Comprehensive validation report formatted and saved successfully",
            "data": {
                "report_file": str(report_file),
                "report_length": len(report),
                "verdict": verdict,
                "files_with_drift": files_with_drift,
                "policy_violations": len(policy_violations),
                "risk_level": risk_level
            }
        }
        
    except Exception as e:
        logger.exception(f"Failed to format report: {e}")
        return {
            "success": False,
            "error": f"Failed to format report: {str(e)}"
        }


@tool
def save_validation_report(
    run_id: str,
    report: str = ""
) -> dict:
    """
    Save validation report to file (Note: format_validation_report now auto-saves).
    
    This tool is kept for backward compatibility but is now optional since
    format_validation_report automatically saves the report.
    
    Args:
        run_id: Validation run ID
        report: Formatted report text (optional, will check for existing file)
        
    Returns:
        Confirmation of save
        
    Example:
        >>> save_validation_report("run_xyz", "## Validation Report...")
    """
    try:
        from pathlib import Path
        report_dir = Path("config_data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"{run_id}_report.md"
        
        # Check if report was already saved by format_validation_report
        if report_file.exists() and not report:
            logger.info(f"Report already saved by format_validation_report: {report_file}")
            return {
                "success": True,
                "message": f"Report already saved for run {run_id}",
                "data": {"report_file": str(report_file)}
            }
        
        # Save report if provided
        if report:
            report_file.write_text(report)
            logger.info(f"Saved report for run {run_id} to {report_file}")
        
        return {
            "success": True,
            "message": f"Report confirmed for run {run_id}",
            "data": {"report_file": str(report_file)}
        }
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return {
            "success": False,
            "error": f"Failed to save report: {str(e)}"
        }


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_supervisor_agent() -> Agent:
    """
    Create and configure the Supervisor Agent.
    
    Returns:
        Configured Strands Agent instance with Bedrock Claude Sonnet model
        
    Example:
        >>> agent = create_supervisor_agent()
        >>> result = agent("Validate configurations")
    """
    # Model configuration - Claude 3.5 Sonnet (smarter for orchestration)
    model = BedrockModel(
        model_id=config.bedrock_model_id,
        region_name=config.aws_region
    )
    
    # Create agent with orchestration tools (simplified to 5 tools)
    agent = Agent(
        name="supervisor",
        description="Orchestrates 2-agent validation workflow (Config Collector + Diff Engine)",
        system_prompt=SYSTEM_PROMPT,
        model=model,
        tools=[
            create_validation_run,
            execute_worker_pipeline,
            aggregate_validation_results,
            format_validation_report,
            save_validation_report
        ]
    )
    
    logger.info("Supervisor Agent created with 2-agent orchestration")
    return agent


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def run_validation(
    project_id: str,
    mr_iid: str,
    repo_url: str,
    golden_branch: str,
    drift_branch: str,
    target_folder: str = ""
) -> dict:
    """
    High-level function to run complete 2-agent file-based validation.
    
    This is the main entry point for running a complete validation using
    file-based communication between agents.
    
    Args:
        project_id: Project identifier (e.g., "myorg/myrepo")
        mr_iid: Merge request or validation ID
        repo_url: GitLab repository URL
        golden_branch: Golden/reference branch name
        drift_branch: Drift/comparison branch name
        target_folder: Optional subfolder to analyze
        
    Returns:
        Dictionary with validation results and file paths
        
    Example:
        >>> result = run_validation(
        ...     "myorg/myrepo", "123",
        ...     "https://gitlab.verizon.com/saja9l7/golden_config.git",
        ...     "gold", "drift"
        ... )
        >>> print(f"Analysis file: {result['diff_analysis_file']}")
        
    Raises:
        Exception: If validation fails
    """
    start_time = datetime.now()
    logger.info(f"Starting 2-agent file-based validation for {mr_iid} in {project_id}")
    
    # Create supervisor agent
    agent = create_supervisor_agent()
    
    # Instruction for supervisor
    instruction = f"""
    Please orchestrate the complete 2-agent file-based validation workflow:
    
    Project: {project_id}
    MR/ID: {mr_iid}
    Repository: {repo_url}
    Golden Branch: {golden_branch}
    Drift Branch: {drift_branch}
    Target Folder: {target_folder or "entire repository"}
    
    Complete workflow:
    1. Create a unique validation run ID
    2. Execute the 2-agent file-based pipeline:
       - Config Collector: Fetch Git diffs, save to drift_analysis file
       - Diff Engine: Read drift_analysis, analyze with AI, save to diff_analysis file
    3. Aggregate results from both agents
    4. Format a comprehensive validation report with:
       - Clear verdict (PASS/FAIL)
       - Summary of drift findings
       - Policy violations
       - Risk assessment
       - Recommendations
    5. Save the validation report to file
    
    Use file paths for communication between agents (no in-memory data transfer).
    Be thorough and ensure all steps complete successfully.
    """
    
    # Execute supervisor
    result = agent(instruction)
    
    # Calculate execution time
    execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    logger.info(f"Validation completed in {execution_time}ms")
    
    # Try to load the latest LLM output and aggregated results for rich UI data
    llm_output_data = {}
    aggregated_data = {}
    
    try:
        # Find latest LLM output (NEW!)
        llm_output_files = sorted(glob.glob("config_data/llm_output/llm_output_*.json"), reverse=True)
        if llm_output_files:
            with open(llm_output_files[0], 'r', encoding='utf-8') as f:
                llm_output_data = json.load(f)
            logger.info(f"Loaded LLM output for UI: {llm_output_files[0]}")
        
        # Find latest aggregated results  
        aggregated_files = sorted(glob.glob("config_data/aggregated_results/aggregated_*.json"), reverse=True)
        if aggregated_files:
            with open(aggregated_files[0], 'r', encoding='utf-8') as f:
                aggregated_data = json.load(f)
            logger.info(f"Loaded aggregated results for UI: {aggregated_files[0]}")
    except Exception as e:
        logger.warning(f"Could not load analysis files for UI: {e}")
    
    # Build comprehensive response with data for UI
    return {
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mr_iid}",
        "verdict": aggregated_data.get("verdict", "COMPLETED"),
        "summary": str(result),
        "execution_time_ms": execution_time,
        "timestamp": datetime.now().isoformat(),
        # Add rich data for UI
        "files_analyzed": aggregated_data.get("files_analyzed", 0),
        "files_compared": aggregated_data.get("files_compared", 0),
        "files_with_drift": aggregated_data.get("files_with_drift", 0),
        "total_deltas": aggregated_data.get("total_deltas", 0),
        "deltas_analyzed": aggregated_data.get("deltas_analyzed", 0),
        "total_clusters": aggregated_data.get("total_clusters", 0),
        "policy_violations_count": aggregated_data.get("policy_violations_count", 0),
        "policy_violations": aggregated_data.get("policy_violations", []),
        "critical_violations": aggregated_data.get("critical_violations", 0),
        "high_violations": aggregated_data.get("high_violations", 0),
        "overall_risk_level": aggregated_data.get("overall_risk_level", "unknown"),
        "risk_assessment": aggregated_data.get("risk_assessment", {}),
        "recommendations": aggregated_data.get("recommendations", []),
        "environment": aggregated_data.get("environment", "unknown"),
        "clusters": aggregated_data.get("clusters", []),
        "analyzed_deltas": aggregated_data.get("analyzed_deltas", []),
        "deltas_with_patches": aggregated_data.get("deltas_with_patches", []),
        "file_paths": aggregated_data.get("file_paths", {}),
        # NEW: LLM output data for UI
        "llm_output": llm_output_data,
        "llm_output_path": llm_output_files[0] if llm_output_files else None
    }


# ============================================================================
# TESTING / DEBUG
# ============================================================================

if __name__ == "__main__":
    """Quick test of the Supervisor Agent."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s'
    )
    
    # Test agent creation
    print("🎯 Testing Supervisor Agent (2-Agent Orchestration)\n")
    
    agent = create_supervisor_agent()
    print(f"✅ Agent created: {agent.name}")
    print(f"   Model: {config.bedrock_model_id}")
    print(f"   Region: {config.aws_region}")
    print(f"   Tools: {len(agent.tool_registry.registry)}")
    
    for tool_name in agent.tool_registry.registry.keys():
        print(f"   - {tool_name}")
    
    print("\n🔧 Supervisor capabilities:")
    print("   - Create validation runs")
    print("   - Execute 2-agent pipeline (Config Collector + Diff Engine)")
    print("   - Aggregate results from both agents")
    print("   - Format validation reports")
    print("   - Save reports to file")
    
    print("\n✅ Supervisor Agent is ready!")
    print("\nTo use:")
    print("  from Agents.Supervisor.supervisor_agent import run_validation")
    print("  result = run_validation('myorg/myrepo', '123', 'feature', 'gold', {}, {})")

