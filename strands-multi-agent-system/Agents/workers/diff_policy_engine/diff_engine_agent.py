"""
Golden Config AI - Diff Policy Engine Agent (Strands Implementation)

This agent performs intelligent configuration drift analysis and policy validation
using pre-processed scrubbed diff content from the GuardrailScrubAgent.

Key Features:
- AI-driven analysis of scrubbed configuration diffs
- Policy compliance validation and recommendations
- Risk assessment and impact analysis
- Remediation planning and recommendations
"""

import asyncio
import json
import logging
import glob
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools import tool

import sys

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

try:
    from shared.config import Config
    from shared.models import TaskRequest, TaskResponse
    from shared.logging_config import get_agent_logger
    logger = get_agent_logger("diff_engine")
except ImportError:
    # Fallback for standalone execution
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("diff_engine")
    
    class Config:
        def __init__(self):
            self.bedrock_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    class TaskRequest:
        pass
    
    class TaskResponse:
        pass


class DiffPolicyEngineAgent(Agent):
    """
    Diff Policy Engine Agent - Configuration Drift Analysis and Policy Validation
    """

    def __init__(self, config: Config = None):
        system_prompt = self._get_system_prompt()
        if config is None:
            config = Config() 

        super().__init__(
            model=BedrockModel(
                model_id=config.bedrock_model_id,
            ),
            system_prompt=system_prompt,
            tools=[
                self.load_scrubbed_analysis,
                self.analyze_delta_with_policy,          # NEW: Policy-aware delta analysis (Phase 3)
                self.analyze_configuration_drift,         # LEGACY: For backwards compatibility
                self.assess_risk_level,
                self.check_policy_violations,
                self.assess_overall_drift_risk,
                self.generate_recommendations
            ]
        )
        self.config = config

    def process_task(self, task: TaskRequest) -> TaskResponse:
        """
        Process a drift analysis task by reading context_bundle file.
        
        This is the main entry point for the agent when called from the supervisor
        or API endpoints. It reads the context_bundle.json file from ConfigCollectorAgent,
        analyzes structured deltas with policy awareness, and saves results to enhanced_analysis file.
        
        Args:
            task: TaskRequest containing task_id, task_type, and parameters
                 Expected parameters:
                 - context_bundle_file: Path to context_bundle.json from ConfigCollector (NEW)
                 OR (for backwards compatibility):
                 - drift_analysis_file: Path to old drift_analysis JSON file
                 
        Returns:
            TaskResponse with enhanced analysis results and output file path
        """
        start_time = time.time()
        try:
            logger.info("=" * 60)
            logger.info(f"🤖 Diff Engine processing task: {task.task_id}")
            logger.info("=" * 60)
            
            params = task.parameters
            
            # NEW: Support both context_bundle_file (new) and drift_analysis_file (backwards compat)
            context_bundle_file = params.get('context_bundle_file') or params.get('drift_analysis_file')
            
            if not context_bundle_file:
                return TaskResponse(
                    task_id=task.task_id,
                    status="failed",
                    result={},
                    error="Missing required parameter: context_bundle_file or drift_analysis_file",
                    processing_time_seconds=time.time() - start_time,
                    metadata={"agent": "diff_policy_engine"}
                )
            
            # Load context bundle from file
            logger.info(f"📂 Loading context bundle from: {context_bundle_file}")
            try:
                with open(context_bundle_file, 'r', encoding='utf-8') as f:
                    context_bundle = json.load(f)
            except FileNotFoundError:
                return TaskResponse(
                    task_id=task.task_id,
                    status="failed",
                    result={},
                    error=f"Context bundle file not found: {context_bundle_file}",
                    processing_time_seconds=time.time() - start_time,
                    metadata={"agent": "diff_policy_engine"}
                )
            except json.JSONDecodeError as e:
                return TaskResponse(
                    task_id=task.task_id,
                    status="failed",
                    result={},
                    error=f"Invalid JSON in context bundle file: {e}",
                    processing_time_seconds=time.time() - start_time,
                    metadata={"agent": "diff_policy_engine"}
                )
            
            # NEW: Extract deltas from context_bundle (structured format)
            # Context bundle has: deltas, file_changes, dependencies, overview, evidence
            deltas = context_bundle.get('deltas', [])
            file_changes = context_bundle.get('file_changes', {})
            dependencies = context_bundle.get('dependencies', {})
            overview = context_bundle.get('overview', {})
            evidence = context_bundle.get('evidence', [])
            
            logger.info(f"📦 Context Bundle loaded:")
            logger.info(f"   Total deltas: {len(deltas)}")
            logger.info(f"   File changes: {len(file_changes.get('modified', []))}")
            logger.info(f"   Dependencies: {len(dependencies)}")
            logger.info(f"   Evidence items: {len(evidence)}")
            
            # Group deltas by category for analysis
            config_deltas = [d for d in deltas if d.get('category') in ['config', 'spring_profile']]
            dep_deltas = [d for d in deltas if d.get('category') == 'dependency']
            code_deltas = [d for d in deltas if d.get('category') in ['code_hunk', 'file']]
            
            logger.info(f"   Config deltas: {len(config_deltas)}")
            logger.info(f"   Dependency deltas: {len(dep_deltas)}")
            logger.info(f"   Code deltas: {len(code_deltas)}")
            
            # Check if there are deltas to analyze
            if not deltas:
                logger.warning("No deltas found in context bundle")
                return TaskResponse(
                    task_id=task.task_id,
                    status="success",
                    result={
                        "context_bundle_file": context_bundle_file,
                        "ai_analysis": {
                            "message": "No deltas to analyze",
                            "overall_risk_level": "none"
                        }
                    },
                    error=None,
                    processing_time_seconds=time.time() - start_time,
                    metadata={"agent": "diff_policy_engine", "deltas_analyzed": 0}
                )
            
            # NEW: Cluster related deltas (Feature #2)
            logger.info(f"\n📦 Clustering related deltas...")
            logger.info("-" * 60)
            clusters = self._cluster_deltas(deltas)
            logger.info(f"✅ Created {len(clusters)} clusters")
            
            logger.info(f"\n🔍 Analyzing deltas with AI (policy-aware)")
            logger.info("-" * 60)
            
            # NEW: Analyze deltas with AI (policy-aware) - BATCHED BY FILE
            all_violations = []
            all_recommendations = []
            risk_scores = []
            analyzed_deltas = []
            
            # Focus on config and dependency deltas (most important)
            all_deltas_to_analyze = config_deltas[:30] + dep_deltas[:10]  # Reduced limit for more reliable JSON
            
            # DEDUPLICATION: Remove duplicate deltas before LLM processing
            logger.info(f"\n🔍 Deduplicating {len(all_deltas_to_analyze)} deltas before LLM analysis...")
            
            # Create unique key for each delta to identify duplicates
            seen_deltas = {}
            deduplicated_deltas = []
            
            for delta in all_deltas_to_analyze:
                # Create unique key: file + key + old_value + new_value + category
                unique_key = f"{delta.get('file', '')}:{delta.get('key', '')}:{delta.get('old_value', '')}:{delta.get('new_value', '')}:{delta.get('category', '')}"
                
                if unique_key not in seen_deltas:
                    seen_deltas[unique_key] = delta
                    deduplicated_deltas.append(delta)
                else:
                    logger.debug(f"   🔄 Skipping duplicate delta: {delta.get('file', '')}:{delta.get('key', '')}")
            
            logger.info(f"   ✅ Deduplicated: {len(all_deltas_to_analyze)} → {len(deduplicated_deltas)} deltas")
            
            # Update the deltas to analyze with deduplicated version
            all_deltas_to_analyze = deduplicated_deltas

            # Group deltas by file and split large files into smaller batches
            deltas_by_file = {}
            for delta in all_deltas_to_analyze:
                file = delta.get('file', 'unknown')
                if file not in deltas_by_file:
                    deltas_by_file[file] = []
                deltas_by_file[file].append(delta)

            # Split large file batches into smaller chunks (max 10 deltas per batch)
            final_batches = []
            for file, file_deltas in deltas_by_file.items():
                if len(file_deltas) <= 10:
                    final_batches.append((file, file_deltas))
                else:
                    # Split into chunks of 10
                    for i in range(0, len(file_deltas), 10):
                        chunk = file_deltas[i:i+10]
                        batch_name = f"{file}_batch_{i//10 + 1}"
                        final_batches.append((batch_name, chunk))

            logger.info(f"📦 Grouped {len(all_deltas_to_analyze)} deltas into {len(final_batches)} batches for analysis")
            
            # NEW: Analyze each batch with LLM format output
            llm_outputs = []
            
            for batch_name, batch_deltas in final_batches:
                logger.info(f"\n  📄 Analyzing {batch_name} ({len(batch_deltas)} deltas)")
                
                # Get environment from overview
                environment = overview.get('environment', 'production')

                try:
                    # Batch analyze with LLM format output (NEW!)
                    llm_format = asyncio.run(self.analyze_file_deltas_batch_llm_format(
                        file=batch_name,
                        deltas=batch_deltas,
                        environment=environment,
                        overview=overview
                    ))
                    
                    llm_outputs.append(llm_format)
                    
                    logger.info(f"     ✅ LLM format: High={len(llm_format.get('high', []))}, "
                               f"Medium={len(llm_format.get('medium', []))}, "
                               f"Low={len(llm_format.get('low', []))}, "
                               f"Allowed={len(llm_format.get('allowed_variance', []))}")
                    
                    # Extract for backward compatibility - infer from bucket (no fields in new format)
                    for item in llm_format.get('high', []):
                        risk_scores.append(75)  # High = 75
                        all_violations.append({
                            'type': 'configuration',
                            'severity': 'high',
                            'description': item.get('why', 'High risk change detected')
                        })
                    
                    for item in llm_format.get('medium', []):
                        risk_scores.append(50)  # Medium = 50
                        all_violations.append({
                            'type': 'configuration',
                            'severity': 'medium',
                            'description': item.get('why', 'Medium risk change detected')
                        })
                    
                    for item in llm_format.get('low', []):
                        risk_scores.append(25)  # Low = 25

                except Exception as e:
                    logger.warning(f"     ❌ LLM format analysis failed for {batch_name}: {e}")
                    # Use fallback categorization
                    llm_format = self._fallback_llm_categorization(batch_deltas, batch_name)
                    llm_outputs.append(llm_format)
            
            # Merge all LLM outputs into single LLM output file (NEW!)
            logger.info(f"\n📦 Generating final LLM output...")
            merged_llm_output = self._merge_llm_outputs(llm_outputs, overview, context_bundle)
            
            # Save LLM output to file (NEW!)
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
            llm_output_dir = PROJECT_ROOT / "config_data" / "llm_output"
            llm_output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            llm_output_file = llm_output_dir / f"llm_output_{timestamp}.json"
            
            with open(llm_output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_llm_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ LLM output saved: {llm_output_file}")
            logger.info(f"   Total items: {len(merged_llm_output['high']) + len(merged_llm_output['medium']) + len(merged_llm_output['low']) + len(merged_llm_output['allowed_variance'])}")
            
            # Create analyzed_deltas for backward compatibility (flatten LLM output)
            analyzed_deltas = []
            risk_level_map = {'high': 'high', 'medium': 'medium', 'low': 'low', 'allowed_variance': 'low'}
            for bucket in ['high', 'medium', 'low', 'allowed_variance']:
                for item in merged_llm_output.get(bucket, []):
                    analyzed_deltas.append({
                        "delta_id": item.get('id'),
                        "file": item.get('file'),
                        "locator": item.get('locator'),
                        "risk_level": risk_level_map[bucket],
                        "verdict": "DRIFT_BLOCKING" if bucket == 'high' else "DRIFT_WARN" if bucket == 'medium' else "NEW_BUILD_OK",
                        "ai_analysis": item
                    })
            
            # Calculate risk distribution for AI assessment
            risk_distribution = {}
            for score in risk_scores:
                if score >= 75:
                    risk_distribution['high'] = risk_distribution.get('high', 0) + 1
                elif score >= 50:
                    risk_distribution['medium'] = risk_distribution.get('medium', 0) + 1
                else:
                    risk_distribution['low'] = risk_distribution.get('low', 0) + 1
            
            # Get environment from overview (already extracted earlier, but keep for safety)
            environment = overview.get('environment', 'production')
            
            # AI-powered overall risk assessment
            logger.info(f"\n🤖 AI-powered overall risk assessment")
            logger.info(f"   Deltas analyzed: {len(analyzed_deltas)}")
            logger.info(f"   Violations found: {len(all_violations)}")
            try:
                ai_risk_assessment = asyncio.run(self.assess_overall_drift_risk(
                    total_files_changed=len(analyzed_deltas),
                    risk_distribution=risk_distribution,
                    files_with_violations=len([d for d in analyzed_deltas if d.get('ai_analysis', {}).get('policy_violations')]),
                    environment=environment
                ))
                overall_risk = ai_risk_assessment.get('overall_risk_level', 'medium')
                risk_factors = ai_risk_assessment.get('risk_factors', [])
                mitigation_strategies = ai_risk_assessment.get('mitigation_strategies', [])
                mitigation_priority = ai_risk_assessment.get('mitigation_priority', 'standard')
                
                logger.info(f"✅ AI assessment: {overall_risk} risk ({mitigation_priority} priority)")
            except Exception as e:
                logger.warning(f"AI overall risk assessment failed, using fallback: {e}")
                # Fallback to simple calculation
                if risk_scores:
                    avg_risk = sum(risk_scores) / len(risk_scores)
                    if avg_risk >= 75:
                        overall_risk = "high"
                    elif avg_risk >= 50:
                        overall_risk = "medium"
                    else:
                        overall_risk = "low"
                else:
                    overall_risk = "low"
                risk_factors = [f"{len(analyzed_deltas)} deltas analyzed"]
                mitigation_strategies = ["Review all changes", "Test in staging"]
                mitigation_priority = "urgent" if overall_risk == "high" else "standard"
            
            # Build enhanced analysis results
            enhanced_analysis = {
                "context_bundle_source": context_bundle_file,  # Source file
                "clusters": clusters,  # NEW: Clustered deltas with root causes
                "ai_policy_analysis": {
                    "total_deltas_analyzed": len(analyzed_deltas),
                    "total_clusters": len(clusters),  # NEW
                    "clustered_deltas": len([item for cluster in clusters for item in cluster.get('items', [])]),  # NEW
                    "policy_violations": all_violations,
                    "overall_risk_level": overall_risk,
                    "risk_assessment": {
                        "overall_risk_level": overall_risk,
                        "average_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                        "risk_distribution": risk_distribution,
                        "risk_factors": risk_factors,
                        "mitigation_priority": mitigation_priority,
                        "mitigation_strategies": mitigation_strategies
                    },
                    "recommendations": [
                        rec.get('action', '') if isinstance(rec, dict) else str(rec)
                        for rec in all_recommendations[:10]
                    ] if all_recommendations else [
                        "Review all configuration changes before deployment",
                        "Test changes in staging environment",
                        "Verify configurations against golden standard"
                    ]
                },
                "analyzed_deltas_with_ai": analyzed_deltas
            }
            
            # Save enhanced analysis to file
            
            PROJECT_ROOT = Path(context_bundle_file).parent.parent.parent  # Up from bundle_xxx to config_data
            ANALYSIS_DIR = PROJECT_ROOT / "enhanced_analysis"
            ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = ANALYSIS_DIR / f"enhanced_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n✅ Diff Engine completed!")
            logger.info(f"   Deltas analyzed: {len(analyzed_deltas)}")
            logger.info(f"   Clusters created: {len(clusters)}")  # NEW
            logger.info(f"   Clustered deltas: {len([item for cluster in clusters for item in cluster.get('items', [])])}")  # NEW
            logger.info(f"   Policy violations: {len(all_violations)}")
            logger.info(f"   Overall risk: {overall_risk}")
            logger.info(f"   Output: {output_file}")
            logger.info("=" * 60)
            
            processing_time = time.time() - start_time
            return TaskResponse(
                task_id=task.task_id,
                status="success",
                result={
                    "enhanced_analysis_file": str(output_file),  # For backward compatibility
                    "llm_output_file": str(llm_output_file),  # NEW: LLM format output
                    "context_bundle_source": context_bundle_file,
                    "summary": {
                        "deltas_analyzed": len(analyzed_deltas),
                        "policy_violations": len(all_violations),
                        "overall_risk": overall_risk,
                        "mitigation_priority": mitigation_priority,
                        "llm_output": {  # NEW: LLM output summary
                            "high": len(merged_llm_output.get('high', [])),
                            "medium": len(merged_llm_output.get('medium', [])),
                            "low": len(merged_llm_output.get('low', [])),
                            "allowed_variance": len(merged_llm_output.get('allowed_variance', []))
                        }
                    }
                },
                error=None,
                processing_time_seconds=processing_time,
                metadata={
                    "agent": "diff_policy_engine",
                    "enhanced_analysis_file": str(output_file),
                    "llm_output_file": str(llm_output_file),  # NEW
                    "deltas_analyzed": len(analyzed_deltas),
                    "risk_level": overall_risk
                }
            )
            
        except Exception as e:
            logger.exception(f"❌ Diff Engine task processing failed: {e}")
            return TaskResponse(
                task_id=task.task_id,
                status="failure",
                result={},
                error=str(e),
                processing_time_seconds=time.time() - start_time,
                metadata={"agent": "diff_policy_engine"}
            )

    def _get_system_prompt(self) -> str:
        """System prompt for the Diff Policy Engine Agent"""
        return """You are the Diff Policy Engine Agent in the Golden Config AI system.

Your primary responsibility is to analyze configuration changes and drift with intelligent insights and policy validation.

CORE CAPABILITIES:
1. Drift Analysis: Detect and classify configuration changes between environments
2. Risk Assessment: Evaluate risk levels (low, medium, high, critical) based on configuration changes
3. Policy Validation: Ensure changes comply with organizational policies and standards
4. Recommendations: Provide actionable recommendations for addressing issues

ANALYSIS FOCUS AREAS:
- Security configuration changes (authentication, encryption, access control)
- Performance impact changes (resource limits, timeouts, caching)
- Compliance adherence (regulatory requirements, internal policies)
- Operational risk assessment (availability, reliability, maintainability)

Always provide clear, actionable insights with specific reasoning for your assessments."""

    def find_latest_file(self, pattern: str) -> Optional[Path]:
        """Find the latest file matching the given pattern."""
        files = glob.glob(pattern)
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        return Path(latest)

    @tool
    def load_scrubbed_analysis(self, scrubbed_analysis_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load scrubbed analysis data from a file.

        Args:
            scrubbed_analysis_file: Path to scrubbed analysis file (optional - will find latest if not provided)

        Returns:
            Loaded scrubbed analysis data as a dictionary.
        """
        PROJECT_ROOT = Path(__file__).resolve()
        for _ in range(6):
            if (PROJECT_ROOT / "config_data").exists():
                break
            if PROJECT_ROOT.parent == PROJECT_ROOT:
                break
            PROJECT_ROOT = PROJECT_ROOT.parent

        if not (PROJECT_ROOT / "config_data").exists():
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
        
        # Ensure PROJECT_ROOT is always a Path object
        if not isinstance(PROJECT_ROOT, Path):
            PROJECT_ROOT = Path(PROJECT_ROOT)

        SCRUBBED_ANALYSIS_DIR = PROJECT_ROOT / "config_data" / "scrubbed_analysis"

        if scrubbed_analysis_file:
            scrubbed_file = Path(scrubbed_analysis_file)
            if not scrubbed_file.is_absolute():
                scrubbed_file = SCRUBBED_ANALYSIS_DIR / scrubbed_analysis_file
        else:
            scrubbed_pattern = str(SCRUBBED_ANALYSIS_DIR / "scrubbed_analysis_*.json")
            scrubbed_file = self.find_latest_file(scrubbed_pattern)

        if not scrubbed_file or not scrubbed_file.exists():
            raise FileNotFoundError("No scrubbed analysis file found")

        with open(scrubbed_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _generate_patch_hint(self, delta_context: dict) -> Optional[Dict[str, str]]:
        """
        Generate copy-pasteable patch hints based on locator type.
        
        Args:
            delta_context: Delta with file, locator, old_value, new_value
            
        Returns:
            Dict with type and content, or None if not applicable
        """
        file = delta_context.get('file', '')
        locator = delta_context.get('locator', {})
        locator_type = locator.get('type', '')
        locator_value = locator.get('value', '')
        old_value = delta_context.get('old_value', '')
        new_value = delta_context.get('new_value', '')
        line_start = locator.get('line_start')
        
        # Extract key from locator value (e.g., "server.yml.ssl.enabled" -> "ssl.enabled")
        if '.' in locator_value:
            parts = locator_value.split('.', 1)
            key_path = parts[1] if len(parts) > 1 else locator_value
        else:
            key_path = locator_value
        
        try:
            # YAML/YML files
            if locator_type == 'yamlpath' or file.endswith(('.yml', '.yaml')):
                # Build YAML structure from key path
                keys = key_path.split('.')
                indent = 0
                lines = []
                
                # Add file header
                if line_start:
                    lines.append(f"# {file} (line {line_start})")
                else:
                    lines.append(f"# {file}")
                
                # Build nested YAML
                for i, key in enumerate(keys):
                    indent_str = "  " * i
                    if i == len(keys) - 1:
                        # Last key - show the value change
                        lines.append(f"{indent_str}{key}: {new_value}    # ← Change to this")
                    else:
                        lines.append(f"{indent_str}{key}:")
                
                return {
                    "type": "yaml_snippet",
                    "content": "\n".join(lines)
                }
            
            # JSON files
            elif locator_type == 'jsonpath' or file.endswith('.json'):
                keys = key_path.split('.')
                lines = []
                
                if line_start:
                    lines.append(f"// {file} (line {line_start})")
                else:
                    lines.append(f"// {file}")
                
                # Build JSON structure
                lines.append("{")
                for i, key in enumerate(keys):
                    indent_str = "  " * (i + 1)
                    if i == len(keys) - 1:
                        # Last key - show value
                        json_value = json.dumps(new_value) if not isinstance(new_value, str) or not new_value.startswith('"') else new_value
                        lines.append(f'{indent_str}"{key}": {json_value}  // ← Change to this')
                    else:
                        lines.append(f'{indent_str}"{key}": {{')
                
                # Close braces
                for i in range(len(keys) - 1, -1, -1):
                    indent_str = "  " * (i + 1)
                    lines.append(f"{indent_str}}}")
                
                return {
                    "type": "json_snippet",
                    "content": "\n".join(lines)
                }
            
            # Properties/INI files
            elif locator_type == 'keypath' or file.endswith(('.properties', '.ini', '.cfg', '.conf')):
                lines = []
                
                if line_start:
                    lines.append(f"# {file} (line {line_start})")
                else:
                    lines.append(f"# {file}")
                
                # Simple key=value format
                lines.append(f"{key_path}={new_value}")
                
                return {
                    "type": "properties_snippet",
                    "content": "\n".join(lines)
                }
            
            # Code files (unidiff format)
            elif locator_type == 'unidiff':
                old_start = locator.get('old_start', 0)
                new_start = locator.get('new_start', 0)
                snippet = delta_context.get('snippet', '')
                
                if snippet:
                    # Use existing snippet from drift.py
                    return {
                        "type": "unified_diff",
                        "content": snippet
                    }
                else:
                    # Build minimal diff
                    lines = [
                        f"--- a/{file}",
                        f"+++ b/{file}",
                        f"@@ -{old_start},1 +{new_start},1 @@",
                        f"- {old_value}",
                        f"+ {new_value}"
                    ]
                    return {
                        "type": "unified_diff",
                        "content": "\n".join(lines)
                    }
            
            # Dependencies
            elif locator_type == 'coord':
                # Dependency coordinates
                coord = locator_value.split(':', 1)
                if len(coord) == 2:
                    ecosystem, package = coord
                    lines = [f"# Update dependency in {file}"]
                    
                    if ecosystem == 'npm':
                        lines.append(f'"{package}": "{new_value}"')
                    elif ecosystem == 'pip':
                        lines.append(f'{package}=={new_value}')
                    elif ecosystem == 'maven':
                        lines.append(f"<version>{new_value}</version>")
                    else:
                        lines.append(f"{package}: {new_value}")
                    
                    return {
                        "type": "dependency_update",
                        "content": "\n".join(lines)
                    }
            
            # Default: generic text format
            else:
                lines = [
                    f"# {file}",
                    f"# Location: {locator_value}",
                    f"",
                    f"Change from: {old_value}",
                    f"Change to:   {new_value}"
                ]
                return {
                    "type": "generic",
                    "content": "\n".join(lines)
                }
                
        except Exception as e:
            logger.warning(f"Failed to generate patch hint for {file}: {e}")
            return None

    def _cluster_deltas(self, deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group related deltas into clusters with root causes.
        
        Clustering strategies:
        1. Same file + similar keys (e.g., all SSL configs in server.yml)
        2. Same policy violation type across files
        3. Same dependency ecosystem changes
        4. Related configuration changes (e.g., database URL + credentials)
        
        Args:
            deltas: List of deltas from context_bundle
            
        Returns:
            List of clusters with root causes and grouped items
        """
        if not deltas:
            return []
        
        logger.info(f"\n🔍 Clustering {len(deltas)} deltas...")
        
        clusters = []
        clustered_delta_ids = set()
        
        # Strategy 1: Group by file + policy violation type
        violation_groups = {}
        for delta in deltas:
            if delta.get('id') in clustered_delta_ids:
                continue
            
            policy = delta.get('policy', {})
            policy_tag = policy.get('tag', 'unknown')
            policy_rule = policy.get('rule', '')
            file = delta.get('file', 'unknown')
            
            # Group invariant breaches by rule + file
            if policy_tag == 'invariant_breach' and policy_rule:
                key = f"{policy_rule}|{file}"
                if key not in violation_groups:
                    violation_groups[key] = []
                violation_groups[key].append(delta)
        
        # Create clusters from violation groups
        for key, group_deltas in violation_groups.items():
            if len(group_deltas) >= 2:  # At least 2 related deltas
                policy_rule, file = key.split('|', 1)
                
                cluster = {
                    "id": f"cluster_violation_{policy_rule}_{hash(file) % 10000}",
                    "type": "policy_violation",
                    "root_cause": f"Policy violation: {policy_rule} in {file}",
                    "items": [d['id'] for d in group_deltas],
                    "file": file,
                    "severity": "critical",
                    "confidence": 0.9,
                    "verdict": "DRIFT_BLOCKING",
                    "deltas": group_deltas
                }
                clusters.append(cluster)
                clustered_delta_ids.update(cluster["items"])
        
        # Strategy 2: Group by similar configuration keys (e.g., SSL/TLS)
        config_pattern_groups = {}
        for delta in deltas:
            if delta.get('id') in clustered_delta_ids:
                continue
            
            if delta.get('category') not in ['config', 'spring_profile']:
                continue
            
            locator_value = delta.get('locator', {}).get('value', '').lower()
            
            # Common patterns to group
            patterns = {
                'ssl_tls': ['ssl', 'tls', 'secure', 'certificate', 'cert', 'https'],
                'database': ['database', 'db', 'datasource', 'jdbc', 'sql'],
                'authentication': ['auth', 'login', 'credential', 'password', 'user', 'jwt'],
                'monitoring': ['log', 'metric', 'monitor', 'health', 'actuator'],
                'cache': ['cache', 'redis', 'memcached', 'ttl'],
                'timeout': ['timeout', 'deadline', 'duration', 'delay'],
                'port': ['port', 'host', 'address', 'endpoint', 'url']
            }
            
            for pattern_name, keywords in patterns.items():
                if any(kw in locator_value for kw in keywords):
                    if pattern_name not in config_pattern_groups:
                        config_pattern_groups[pattern_name] = []
                    config_pattern_groups[pattern_name].append(delta)
                    break
        
        # Create clusters from pattern groups
        pattern_descriptions = {
            'ssl_tls': 'SSL/TLS configuration changes',
            'database': 'Database configuration changes',
            'authentication': 'Authentication configuration changes',
            'monitoring': 'Monitoring and logging changes',
            'cache': 'Caching configuration changes',
            'timeout': 'Timeout and performance settings',
            'port': 'Network endpoint configuration changes'
        }
        
        for pattern_name, group_deltas in config_pattern_groups.items():
            if len(group_deltas) >= 3:  # At least 3 related config changes
                # Calculate severity based on policy tags
                has_breach = any(d.get('policy', {}).get('tag') == 'invariant_breach' for d in group_deltas)
                severity = "critical" if has_breach else "high"
                verdict = "DRIFT_BLOCKING" if has_breach else "DRIFT_WARN"
                
                files = list(set(d.get('file', '') for d in group_deltas))
                file_summary = f"{len(files)} file(s)" if len(files) > 1 else files[0]
                
                cluster = {
                    "id": f"cluster_pattern_{pattern_name}",
                    "type": "configuration_pattern",
                    "root_cause": f"{pattern_descriptions[pattern_name]} across {file_summary}",
                    "items": [d['id'] for d in group_deltas],
                    "files": files,
                    "pattern": pattern_name,
                    "severity": severity,
                    "confidence": 0.85,
                    "verdict": verdict,
                    "deltas": group_deltas
                }
                clusters.append(cluster)
                clustered_delta_ids.update(cluster["items"])
        
        # Strategy 3: Group dependency changes by ecosystem
        dep_groups = {}
        for delta in deltas:
            if delta.get('id') in clustered_delta_ids:
                continue
            
            if delta.get('category') != 'dependency':
                continue
            
            locator = delta.get('locator', {})
            locator_value = locator.get('value', '')
            
            # Extract ecosystem (npm:, pip:, maven:)
            if ':' in locator_value:
                ecosystem = locator_value.split(':', 1)[0]
                if ecosystem not in dep_groups:
                    dep_groups[ecosystem] = []
                dep_groups[ecosystem].append(delta)
        
        # Create clusters from dependency groups
        for ecosystem, group_deltas in dep_groups.items():
            if len(group_deltas) >= 3:  # At least 3 dependency changes
                cluster = {
                    "id": f"cluster_deps_{ecosystem}",
                    "type": "dependency_update",
                    "root_cause": f"Multiple {ecosystem} dependency updates",
                    "items": [d['id'] for d in group_deltas],
                    "ecosystem": ecosystem,
                    "severity": "medium",
                    "confidence": 0.8,
                    "verdict": "DRIFT_WARN",
                    "deltas": group_deltas
                }
                clusters.append(cluster)
                clustered_delta_ids.update(cluster["items"])
        
        logger.info(f"✅ Created {len(clusters)} clusters covering {len(clustered_delta_ids)} deltas")
        for cluster in clusters:
            logger.info(f"   • {cluster['id']}: {cluster['root_cause']} ({len(cluster['items'])} items)")
        
        return clusters

    def _format_pinpoint_location(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create structured pinpoint location reference for easy navigation.
        
        Converts locator information into a standardized pinpoint format
        that includes file, line, column, and navigation hints.
        
        Args:
            delta: Delta with locator information
            
        Returns:
            Structured pinpoint location with navigation details
        """
        file = delta.get('file', 'unknown')
        locator = delta.get('locator', {})
        locator_type = locator.get('type', 'unknown')
        locator_value = locator.get('value', '')
        line_start = locator.get('line_start')
        line_end = locator.get('line_end')
        column_start = locator.get('column_start')
        column_end = locator.get('column_end')
        
        # Build base pinpoint structure
        pinpoint = {
            "file": file,
            "type": locator_type,
            "path": locator_value,
            "navigation": {},
            "quick_access": {}
        }
        
        # Add line/column information
        if line_start is not None:
            pinpoint["line"] = {
                "start": line_start,
                "end": line_end if line_end else line_start,
                "range": f"{line_start}" if line_end == line_start else f"{line_start}-{line_end}"
            }
        
        if column_start is not None:
            pinpoint["column"] = {
                "start": column_start,
                "end": column_end if column_end else column_start,
                "range": f"{column_start}" if column_end == column_start else f"{column_start}-{column_end}"
            }
        
        # Add navigation hints based on locator type
        if locator_type == 'yamlpath':
            pinpoint["navigation"] = {
                "type": "YAML Path",
                "description": f"YAML key: {locator_value}",
                "search_hint": f"Search for: {locator_value.split('.')[-1]}",
                "vs_code_command": f"Ctrl+Shift+F → '{locator_value.split('.')[-1]}'",
                "vim_command": f"/{locator_value.split('.')[-1]}"
            }
            pinpoint["quick_access"] = {
                "jump_to_key": locator_value.split('.')[-1],
                "full_path": locator_value,
                "parent_keys": locator_value.split('.')[:-1]
            }
            
        elif locator_type == 'jsonpath':
            pinpoint["navigation"] = {
                "type": "JSON Path",
                "description": f"JSON property: {locator_value}",
                "search_hint": f"Search for: {locator_value.split('.')[-1]}",
                "vs_code_command": f"Ctrl+Shift+F → '{locator_value.split('.')[-1]}'",
                "vim_command": f"/{locator_value.split('.')[-1]}"
            }
            pinpoint["quick_access"] = {
                "jump_to_key": locator_value.split('.')[-1],
                "full_path": locator_value,
                "parent_keys": locator_value.split('.')[:-1]
            }
            
        elif locator_type == 'unidiff':
            pinpoint["navigation"] = {
                "type": "Unified Diff",
                "description": f"Code change at line {line_start}",
                "search_hint": f"Go to line {line_start}",
                "vs_code_command": f"Ctrl+G → {line_start}",
                "vim_command": f":{line_start}"
            }
            pinpoint["quick_access"] = {
                "jump_to_line": line_start,
                "context_lines": 3,
                "file_section": "code_change"
            }
            
        elif locator_type == 'keypath':
            pinpoint["navigation"] = {
                "type": "Key Path",
                "description": f"Configuration key: {locator_value}",
                "search_hint": f"Search for: {locator_value}",
                "vs_code_command": f"Ctrl+Shift+F → '{locator_value}'",
                "vim_command": f"/{locator_value}"
            }
            pinpoint["quick_access"] = {
                "jump_to_key": locator_value,
                "key_pattern": locator_value
            }
            
        elif locator_type == 'coord':
            pinpoint["navigation"] = {
                "type": "Coordinates",
                "description": f"Position: line {line_start}, column {column_start}",
                "search_hint": f"Go to {line_start}:{column_start}",
                "vs_code_command": f"Ctrl+G → {line_start}:{column_start}",
                "vim_command": f":{line_start}"
            }
            pinpoint["quick_access"] = {
                "jump_to_line": line_start,
                "jump_to_column": column_start,
                "precise_location": True
            }
            
        elif locator_type == 'file':
            pinpoint["navigation"] = {
                "type": "File",
                "description": f"File: {file}",
                "search_hint": f"Open file: {file}",
                "vs_code_command": f"Ctrl+P → '{file}'",
                "vim_command": f":e {file}"
            }
            pinpoint["quick_access"] = {
                "file_path": file,
                "file_name": file.split('/')[-1]
            }
            
        else:
            # Generic fallback
            pinpoint["navigation"] = {
                "type": "Generic",
                "description": f"Location in {file}",
                "search_hint": f"Search in {file}",
                "vs_code_command": f"Ctrl+P → '{file}'",
                "vim_command": f":e {file}"
            }
            pinpoint["quick_access"] = {
                "file_path": file,
                "search_term": locator_value
            }
        
        # Add IDE-specific quick links
        pinpoint["ide_links"] = {
            "vs_code": f"vscode://file/{Path(file).absolute()}:{line_start or 1}:{column_start or 1}",
            "intellij": f"idea://open?file={Path(file).absolute()}&line={line_start or 1}&column={column_start or 1}",
            "sublime": f"subl://{Path(file).absolute()}:{line_start or 1}:{column_start or 1}"
        }
        
        # Add copy-pasteable location string
        if line_start:
            pinpoint["location_string"] = f"{file}:{line_start}"
            if column_start:
                pinpoint["location_string"] += f":{column_start}"
        else:
            pinpoint["location_string"] = file
            
        # Add human-readable description
        pinpoint["human_readable"] = self._create_human_readable_location(pinpoint)
        
        return pinpoint
    
    def _create_human_readable_location(self, pinpoint: Dict[str, Any]) -> str:
        """Create a human-readable location description."""
        file = pinpoint.get('file', 'unknown')
        nav_type = pinpoint.get('navigation', {}).get('type', 'Location')
        
        if 'line' in pinpoint:
            line_info = pinpoint['line']
            if 'column' in pinpoint:
                col_info = pinpoint['column']
                return f"{file} at line {line_info['range']}, column {col_info['range']} ({nav_type})"
            else:
                return f"{file} at line {line_info['range']} ({nav_type})"
        else:
            return f"{file} ({nav_type})"

    def _check_evidence_requirements(self, delta: Dict[str, Any], evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if delta changes meet evidence/approval requirements.
        
        Validates against:
        1. Required approval tickets for specific changes
        2. Required flags/environment variables
        3. Required documentation
        4. Required testing evidence
        5. Required security reviews
        
        Args:
            delta: Delta with change information
            evidence_list: List of evidence items from context_bundle
            
        Returns:
            Evidence validation result with requirements and status
        """
        file = delta.get('file', 'unknown')
        locator = delta.get('locator', {})
        locator_value = locator.get('value', '')
        old_value = delta.get('old')
        new_value = delta.get('new')
        category = delta.get('category', 'unknown')
        
        evidence_check = {
            "file": file,
            "location": locator_value,
            "requirements": [],
            "evidence_found": [],
            "evidence_missing": [],
            "approval_status": "unknown",
            "compliance_score": 0.0,
            "validation_summary": ""
        }
        
        # Define evidence requirements based on change type and location
        requirements = []
        
        # 1. Security-related changes require security approval
        if any(keyword in locator_value.lower() for keyword in ['ssl', 'tls', 'cert', 'auth', 'password', 'secret', 'key', 'token', 'jwt']):
            requirements.append({
                "type": "security_approval",
                "description": "Security-related change requires security team approval",
                "required": True,
                "ticket_type": "security_review",
                "priority": "high"
            })
        
        # 2. Database changes require DBA approval
        if any(keyword in locator_value.lower() for keyword in ['database', 'db', 'jdbc', 'sql', 'connection', 'datasource']):
            requirements.append({
                "type": "dba_approval", 
                "description": "Database change requires DBA approval",
                "required": True,
                "ticket_type": "dba_review",
                "priority": "high"
            })
        
        # 3. Production environment changes require change management
        if any(keyword in locator_value.lower() for keyword in ['prod', 'production', 'live', 'main']):
            requirements.append({
                "type": "change_management",
                "description": "Production change requires change management ticket",
                "required": True,
                "ticket_type": "change_request",
                "priority": "critical"
            })
        
        # 4. Configuration changes require documentation
        if category in ['config', 'spring_profile']:
            requirements.append({
                "type": "documentation",
                "description": "Configuration change requires documentation update",
                "required": True,
                "ticket_type": "doc_update",
                "priority": "medium"
            })
        
        # 5. Dependency updates require security scan
        if category == 'dependency':
            requirements.append({
                "type": "security_scan",
                "description": "Dependency update requires security vulnerability scan",
                "required": True,
                "ticket_type": "security_scan",
                "priority": "high"
            })
        
        # 6. Performance-related changes require load testing
        if any(keyword in locator_value.lower() for keyword in ['timeout', 'pool', 'cache', 'performance', 'memory', 'cpu']):
            requirements.append({
                "type": "performance_testing",
                "description": "Performance change requires load testing evidence",
                "required": True,
                "ticket_type": "performance_test",
                "priority": "medium"
            })
        
        # 7. Network/port changes require network approval
        if any(keyword in locator_value.lower() for keyword in ['port', 'host', 'endpoint', 'url', 'network']):
            requirements.append({
                "type": "network_approval",
                "description": "Network change requires network team approval",
                "required": True,
                "ticket_type": "network_review",
                "priority": "high"
            })
        
        # 8. Critical system changes require architecture review
        if any(keyword in locator_value.lower() for keyword in ['core', 'critical', 'system', 'framework', 'platform']):
            requirements.append({
                "type": "architecture_review",
                "description": "Critical system change requires architecture team review",
                "required": True,
                "ticket_type": "arch_review",
                "priority": "critical"
            })
        
        evidence_check["requirements"] = requirements
        
        # Check evidence against requirements
        evidence_found = []
        evidence_missing = []
        
        for req in requirements:
            req_type = req["type"]
            found = False
            
            # Look for matching evidence
            for evidence in evidence_list:
                evidence_type = evidence.get('type', '').lower()
                evidence_description = evidence.get('description', '').lower()
                evidence_tags = evidence.get('tags', [])
                
                # Match evidence type to requirement
                if (req_type == "security_approval" and 
                    (evidence_type in ['security', 'approval', 'review'] or 
                     'security' in evidence_description or 'approval' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "dba_approval" and 
                      (evidence_type in ['database', 'dba', 'approval'] or
                       'database' in evidence_description or 'dba' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "change_management" and 
                      (evidence_type in ['change', 'ticket', 'request'] or
                       'change' in evidence_description or 'ticket' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "documentation" and 
                      (evidence_type in ['doc', 'documentation', 'wiki'] or
                       'doc' in evidence_description or 'wiki' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "security_scan" and 
                      (evidence_type in ['scan', 'vulnerability', 'security'] or
                       'scan' in evidence_description or 'vulnerability' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "performance_testing" and 
                      (evidence_type in ['test', 'performance', 'load'] or
                       'test' in evidence_description or 'performance' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "network_approval" and 
                      (evidence_type in ['network', 'infrastructure'] or
                       'network' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
                    
                elif (req_type == "architecture_review" and 
                      (evidence_type in ['architecture', 'review', 'design'] or
                       'architecture' in evidence_description or 'design' in evidence_description)):
                    found = True
                    evidence_found.append({
                        "requirement": req_type,
                        "evidence_id": evidence.get('id', 'unknown'),
                        "evidence_type": evidence_type,
                        "description": evidence.get('description', ''),
                        "status": "found"
                    })
            
            if not found:
                evidence_missing.append({
                    "requirement": req_type,
                    "description": req["description"],
                    "ticket_type": req["ticket_type"],
                    "priority": req["priority"],
                    "status": "missing"
                })
        
        evidence_check["evidence_found"] = evidence_found
        evidence_check["evidence_missing"] = evidence_missing
        
        # Calculate compliance score
        total_requirements = len(requirements)
        found_requirements = len(evidence_found)
        compliance_score = found_requirements / total_requirements if total_requirements > 0 else 1.0
        evidence_check["compliance_score"] = compliance_score
        
        # Determine approval status
        if compliance_score == 1.0:
            approval_status = "approved"
        elif compliance_score >= 0.7:
            approval_status = "partial_approval"
        elif compliance_score >= 0.3:
            approval_status = "pending_review"
        else:
            approval_status = "rejected"
        
        evidence_check["approval_status"] = approval_status
        
        # Generate validation summary
        if compliance_score == 1.0:
            evidence_check["validation_summary"] = f"✅ All evidence requirements met ({found_requirements}/{total_requirements})"
        elif compliance_score >= 0.7:
            evidence_check["validation_summary"] = f"⚠️ Partial evidence found ({found_requirements}/{total_requirements}) - Review required"
        elif compliance_score >= 0.3:
            evidence_check["validation_summary"] = f"🔄 Evidence pending ({found_requirements}/{total_requirements}) - Additional approvals needed"
        else:
            evidence_check["validation_summary"] = f"❌ Insufficient evidence ({found_requirements}/{total_requirements}) - Change blocked"
        
        return evidence_check

    @tool
    async def analyze_delta_with_policy(self, 
                                        delta_context: dict,
                                        environment: str = "production") -> dict:
        """
        Analyze a single delta with policy awareness.
        
        NEW: This method works with structured deltas from context_bundle.json
        and uses pre-evaluated policy tags for intelligent analysis.
        
        Args:
            delta_context: Dict containing:
                - delta_id: Unique identifier for this delta
                - file: File path
                - category: Delta category (config, dependency, code_hunk)
                - locator: Precise locator (yamlpath, jsonpath, unidiff, etc.)
                - old_value: Previous value
                - new_value: New value  
                - policy_tag: Pre-evaluated policy tag (invariant_breach, allowed_variance, suspect)
                - policy_rule: Policy rule name if matched
            environment: Target environment (production, staging, development)
            
        Returns:
            Dict with risk_level, verdict, policy_violations, recommendations, confidence, rationale, patch_hint
        """
        file = delta_context.get('file', 'unknown')
        locator = delta_context.get('locator', {})
        old_value = delta_context.get('old_value', '')
        new_value = delta_context.get('new_value', '')
        policy_tag = delta_context.get('policy_tag', 'unknown')
        policy_rule = delta_context.get('policy_rule', '')
        
        # Generate pinpoint location reference (Feature #3)
        pinpoint = self._format_pinpoint_location(delta_context)
        
        # Generate patch hint for this delta
        patch_hint = self._generate_patch_hint(delta_context)
        
        # Check evidence requirements (Feature #4) - Note: evidence_list would come from context_bundle
        # For now, we'll use an empty list as evidence checking requires evidence data
        evidence_list = []  # TODO: Extract from context_bundle when available
        evidence_check = self._check_evidence_requirements(delta_context, evidence_list)
        
        # Build policy-aware prompt from context-generator's adjudicator.md
        prompt = f"""You are a policy-aware configuration drift analyst.

### Delta to Analyze:

**File:** {file}
**Locator:** {locator.get('type', 'unknown')}: {locator.get('value', '')}
{f"**Line:** {locator.get('line_start')}" if 'line_start' in locator else ""}

**Change:**
- Old Value: `{old_value}`
- New Value: `{new_value}`

### Pre-Evaluated Policy Status:

**Policy Tag:** {policy_tag}
**Policy Rule:** {policy_rule or "None"}

**Policy Interpretation:**
- `invariant_breach` = CRITICAL policy violation detected by explicit rules
- `allowed_variance` = Expected environment-specific difference (likely acceptable)
- `suspect` = Requires AI analysis to determine impact

### Your Task:

Based on the policy tag and the specific change:

1. **If policy_tag = "invariant_breach":**
   - This IS a policy violation
   - Mark as HIGH or CRITICAL severity
   - Explain why this violates organizational policy
   - Verdict should be DRIFT_BLOCKING

2. **If policy_tag = "allowed_variance":**
   - This is likely acceptable (e.g., dev/staging configs differ from prod)
   - But still assess if there are any concerns
   - Risk level: LOW
   - Verdict: NO_DRIFT or NEW_BUILD_OK

3. **If policy_tag = "suspect":**
   - Analyze the change
   - Determine if it poses risk
   - Consider security, operational, compliance impact
   - Verdict based on analysis

### Environment Context:
**Target:** {environment}
**Strictness:** {"STRICT - Zero tolerance for security issues" if environment == "production" else "MODERATE - Review required but flexible"}

### Output (JSON):

{{
  "risk_level": "low|medium|high|critical",
  "verdict": "NO_DRIFT|NEW_BUILD_OK|DRIFT_WARN|DRIFT_BLOCKING",
  "policy_violations": [
    {{
      "rule": "{policy_rule or 'policy_check'}",
      "type": "security|operational|compliance",
      "severity": "low|medium|high|critical",
      "description": "What was violated",
      "recommendation": "How to fix"
    }}
  ],
  "recommendations": [
    {{
      "priority": "immediate|high|medium|low",
      "action": "Specific action to take",
      "rationale": "Why this matters"
    }}
  ],
  "confidence": 0.0-1.0,
  "rationale": "Brief explanation of your assessment"
}}

**Remember:** Be strict for production, considerate for staging/dev. If invariant_breach, mark as critical."""

        # Run AI analysis using the model directly
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        ai_response = ""
        async for event in self.model.stream(messages, max_tokens=800):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    ai_response += delta["text"]
        
        # Parse AI response (expecting JSON)
        try:
            # Extract JSON from response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", ai_response, 0)
            # Add patch_hint to result
            if patch_hint:
                result["patch_hint"] = patch_hint
            result["pinpoint"] = pinpoint  # NEW: Add pinpoint location
            result["evidence_check"] = evidence_check  # NEW: Add evidence checking
            return result
        except json.JSONDecodeError:
            # Fallback if AI doesn't return valid JSON
            logger.warning(f"AI returned non-JSON response for {file}")
            # Determine fallback based on policy_tag
            if policy_tag == "invariant_breach":
                fallback_result = {
                    "risk_level": "critical",
                    "verdict": "DRIFT_BLOCKING",
                    "policy_violations": [{
                        "rule": policy_rule or "policy_breach",
                        "type": "policy",
                        "severity": "critical",
                        "description": f"Policy violation: {policy_rule}",
                        "recommendation": "Revert this change or get approval"
                    }],
                    "recommendations": [],
                    "confidence": 0.9,
                    "rationale": "Explicit policy rule violated"
                }
            elif policy_tag == "allowed_variance":
                fallback_result = {
                    "risk_level": "low",
                    "verdict": "NEW_BUILD_OK",
                    "policy_violations": [],
                    "recommendations": [],
                    "confidence": 0.8,
                    "rationale": "Environment-specific configuration difference"
                }
            else:
                fallback_result = {
                    "risk_level": "medium",
                    "verdict": "DRIFT_WARN",
                    "policy_violations": [],
                    "recommendations": [],
                    "confidence": 0.5,
                    "rationale": "AI analysis failed, requires manual review"
                }
            
            # Add patch_hint to fallback result
            if patch_hint:
                fallback_result["patch_hint"] = patch_hint
            fallback_result["pinpoint"] = pinpoint  # NEW: Add pinpoint location
            fallback_result["evidence_check"] = evidence_check  # NEW: Add evidence checking
            
            return fallback_result

    @tool
    async def analyze_file_deltas_batch(self,
                                        file: str,
                                        deltas: list,
                                        environment: str = "production",
                                        overview: dict = None) -> dict:
        """
        Batch analyze ALL deltas in a single file with one AI call.
        
        This is much more efficient than analyzing deltas one-by-one:
        - 1 AI call per file instead of 1 per delta
        - AI sees full context of all changes in the file
        - Better relationship detection between changes
        
        Args:
            file: File path
            deltas: List of all deltas in this file
            environment: Target environment
            overview: Repository overview context
            
        Returns:
            Dict with delta_analyses (list of analysis results for each delta)
        """
        # Build comprehensive context for all deltas in this file
        deltas_summary = []
        for delta in deltas:
            deltas_summary.append({
                "delta_id": delta.get('id', 'unknown'),
                "locator": f"{delta.get('locator', {}).get('type')}: {delta.get('locator', {}).get('value')}",
                "old_value": str(delta.get('old')) if delta.get('old') is not None else "null",
                "new_value": str(delta.get('new')) if delta.get('new') is not None else "null",
                "policy_tag": delta.get('policy', {}).get('tag', 'unknown'),
                "policy_rule": delta.get('policy', {}).get('rule', '')
            })
        
        # Get policy context for better analysis
        policies = overview.get('policies', {}) if overview else {}
        
        # Build structured batch analysis prompt
        prompt = f"""You are a configuration drift analysis expert. Analyze {len(deltas)} configuration changes in file "{file}" for environment "{environment}".

CONTEXT:
- Environment: {environment}
- File Type: Configuration file
- Analysis Scope: Individual and collective impact assessment

POLICY GUIDELINES:
- Security: Block credential changes, validate authentication settings
- Operational: Warn on port/endpoint changes, flag disabled features
- Compliance: Ensure configuration standards are maintained
- Performance: Identify changes that may affect system performance

ANALYSIS REQUIREMENTS:
1. Examine each change individually and collectively
2. Identify policy violations, security risks, operational impacts
3. Provide specific recommendations and patch hints
4. Assess overall file risk level
5. Consider relationships between changes

CHANGES TO ANALYZE:
"""
        
        # Add each delta with clear structure
        for i, d in enumerate(deltas_summary, 1):
            prompt += f"""
CHANGE #{i}:
- ID: {d['delta_id']}
- Location: {d['locator']}
- Old Value: {d['old_value']}
- New Value: {d['new_value']}
- Policy: {d['policy_tag']}
- Rule: {d['policy_rule'] or 'none'}

"""

        prompt += f"""
OUTPUT FORMAT - Respond with ONLY this JSON structure (no other text):

{{
  "file_risk_level": "low",
  "file_verdict": "DRIFT_WARN",
  "delta_analyses": [
"""

        # Add template for each delta
        for i, d in enumerate(deltas_summary, 1):
            prompt += f"""    {{
      "delta_id": "{d['delta_id']}",
      "risk_level": "medium",
      "verdict": "DRIFT_WARN",
      "policy_violations": [],
      "recommendations": ["Review this change"],
      "confidence": 0.8,
      "rationale": "Configuration change detected",
      "patch_hint": {{
        "type": "yaml",
        "content": "corrected_value_here"
      }}
    }}"""
            if i < len(deltas_summary):
                prompt += ","
            prompt += "\n"

        prompt += """  ]
}

RISK LEVEL GUIDELINES:
- LOW: Minor config changes, non-critical settings
- MEDIUM: Network changes, feature toggles, non-sensitive settings
- HIGH: Authentication changes, security settings, critical paths
- CRITICAL: Credential exposure, security bypasses, breaking changes

VERDICT GUIDELINES:
- NO_DRIFT: No significant changes or improvements only
- NEW_BUILD_OK: New features or enhancements that are safe
- DRIFT_WARN: Changes that need review but aren't blocking
- DRIFT_BLOCKING: Critical changes that should block deployment

IMPORTANT: 
- Use ONLY the exact JSON structure above
- Fill in appropriate values for each delta
- Ensure all brackets and commas are correct
- Do not add any text outside the JSON
- Be thorough in your analysis but concise in descriptions"""
        
        # Call AI with batch prompt
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        ai_response = ""
        async for event in self.model.stream(messages, max_tokens=4000):  # Increased for batch
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    ai_response += delta["text"]
        
        # Extract JSON from response with robust parsing
        result = self._parse_ai_json_response(ai_response)
        return result

    @tool
    async def analyze_file_deltas_batch_llm_format(self,
                                                    file: str,
                                                    deltas: list,
                                                    environment: str = "production",
                                                    overview: dict = None) -> dict:
        """
        Batch analyze ALL deltas in a single file with one AI call - LLM OUTPUT FORMAT.
        
        This method returns the LLM output format DIRECTLY (high/medium/low/allowed_variance)
        instead of the nested delta_analyses structure. This eliminates post-processing.
        
        Args:
            file: File path
            deltas: List of all deltas in this file
            environment: Target environment (production, staging, dev, qa)
            overview: Repository overview context
        
        Returns:
            Dict with high, medium, low, allowed_variance arrays (LLM format)
        """
        from .prompts.llm_format_prompt import build_llm_format_prompt, validate_llm_output
        
        logger.info(f"     📋 Building LLM format prompt for {len(deltas)} deltas...")
        
        # Get policies from overview
        policies = overview.get('policies', {}) if overview else {}
        
        # Build prompt using template
        prompt = build_llm_format_prompt(
            file=file,
            deltas=deltas,
            environment=environment,
            policies=policies
        )
        
        logger.info(f"     🤖 Calling AI for LLM format analysis (max_tokens=8000)...")
        
        # Call AI with LLM format prompt
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        ai_response = ""
        async for event in self.model.stream(messages, max_tokens=8000):  # Increased for larger output
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    ai_response += delta["text"]
        
        logger.info(f"     ✅ Received AI response ({len(ai_response)} chars)")
        
        # Parse JSON response with robust error handling
        try:
            result = self._parse_ai_json_response(ai_response)
            
            # Validate LLM output structure
            if not validate_llm_output(result):
                logger.warning(f"     ⚠️ LLM output validation failed, missing required fields")
                raise ValueError("Invalid LLM output structure")
            
            logger.info(f"     ✅ Valid LLM format: High={len(result.get('high', []))}, "
                       f"Medium={len(result.get('medium', []))}, "
                       f"Low={len(result.get('low', []))}, "
                       f"Allowed={len(result.get('allowed_variance', []))}")
            
            return result
            
        except Exception as e:
            logger.error(f"     ❌ Failed to parse LLM output: {e}")
            logger.error(f"     Raw response (first 500 chars): {ai_response[:500]}")
            raise

    def _extract_batch_results(self, batch_analysis, batch_deltas, batch_name, 
                              analyzed_deltas, risk_scores, all_violations, all_recommendations):
        """Extract results from batch analysis with robust error handling"""
        try:
            delta_analyses = batch_analysis.get('delta_analyses', [])
            if not delta_analyses:
                logger.warning(f"     ⚠️ No delta_analyses found in batch response")
                return
            
            for delta_result in delta_analyses:
                delta_id = delta_result.get('delta_id')
                if not delta_id:
                    logger.warning(f"     ⚠️ Skipping delta result with no delta_id")
                    continue

                # Find original delta
                original_delta = next((d for d in batch_deltas if d.get('id') == delta_id), None)
                if not original_delta:
                    logger.warning(f"     ⚠️ Could not find original delta for {delta_id}")
                    continue

                risk_level = delta_result.get('risk_level', 'medium')
                verdict = delta_result.get('verdict', 'DRIFT_WARN')

                risk_score_map = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100}
                risk_scores.append(risk_score_map.get(risk_level, 50))

                violations = delta_result.get('policy_violations', [])
                all_violations.extend(violations)

                recommendations = delta_result.get('recommendations', [])
                all_recommendations.extend(recommendations)

                # Store analyzed delta
                analyzed_deltas.append({
                    "delta_id": delta_id,
                    "file": batch_name,
                    "locator": original_delta.get('locator', {}),
                    "policy_tag": original_delta.get('policy', {}).get('tag', 'unknown'),
                    "old_value": original_delta.get('old'),
                    "new_value": original_delta.get('new'),
                    "ai_analysis": delta_result,
                    "risk_level": risk_level,
                    "verdict": verdict,
                    "policy_violations": violations,
                    "recommendations": recommendations,
                    "patch_hint": delta_result.get('patch_hint'),
                    "pinpoint": delta_result.get('pinpoint'),
                    "evidence_check": delta_result.get('evidence_check')
                })
                
        except Exception as e:
            logger.error(f"     ❌ Error extracting batch results: {e}")
            # If we can't extract results, fall back to individual analysis
            self._fallback_individual_analysis(
                batch_deltas, batch_name, analyzed_deltas, risk_scores, 
                all_violations, all_recommendations, f"Result extraction failed: {e}"
            )

    def _fallback_individual_analysis(self, batch_deltas, batch_name, analyzed_deltas, 
                                    risk_scores, all_violations, all_recommendations, error_msg):
        """Fallback: analyze each delta individually with simple rule-based logic"""
        logger.info(f"     🔄 Using fallback individual analysis for {len(batch_deltas)} deltas")
        
        for delta in batch_deltas:
            # Simple rule-based risk assessment
            risk_level, verdict, violations, recommendations = self._simple_risk_assessment(delta)
            
            risk_score_map = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100}
            risk_scores.append(risk_score_map.get(risk_level, 50))
            
            all_violations.extend(violations)
            all_recommendations.extend(recommendations)
            
            # Store analyzed delta
            analyzed_deltas.append({
                "delta_id": delta.get('id', 'unknown'),
                "file": batch_name,
                "locator": delta.get('locator', {}),
                "policy_tag": delta.get('policy', {}).get('tag', 'unknown'),
                "old_value": delta.get('old'),
                "new_value": delta.get('new'),
                "ai_analysis": {
                    "error": error_msg,
                    "fallback_analysis": True,
                    "risk_level": risk_level,
                    "verdict": verdict
                },
                "risk_level": risk_level,
                "verdict": verdict,
                "policy_violations": violations,
                "recommendations": recommendations,
                "patch_hint": None,
                "pinpoint": None,
                "evidence_check": None
            })

    def _simple_risk_assessment(self, delta):
        """Simple rule-based risk assessment when AI analysis fails"""
        old_val = str(delta.get('old', '')).lower()
        new_val = str(delta.get('new', '')).lower()
        policy_tag = delta.get('policy', {}).get('tag', '').lower()
        
        # Simple rules for common high-risk patterns
        if any(keyword in new_val for keyword in ['password', 'secret', 'key', 'token']):
            if old_val != new_val:  # Only if actually changed
                return 'high', 'DRIFT_BLOCKING', ['Sensitive data exposure'], ['Review credential changes']
        
        if any(keyword in new_val for keyword in ['port', 'host', 'url', 'endpoint']):
            return 'medium', 'DRIFT_WARN', ['Network configuration change'], ['Verify network settings']
        
        if 'disabled' in new_val or 'false' in new_val:
            return 'medium', 'DRIFT_WARN', ['Feature disabled'], ['Verify this is intentional']
        
        # Default assessment
        return 'low', 'DRIFT_WARN', [], ['Review configuration change']

    def _fallback_llm_categorization(self, deltas: list, file: str) -> dict:
        """
        Fallback: Simple rule-based categorization in EXACT LLM format when AI fails.
        
        Matches LLM_output.json format exactly:
        - high/medium/low: id, file, locator, old, new, why, remediation
        - allowed_variance: id, file, locator, old, new, rationale
        
        Args:
            deltas: List of deltas to categorize
            file: File path
        
        Returns:
            Dict in LLM format (high/medium/low/allowed_variance) - EXACT MATCH
        """
        logger.info(f"     🔄 Using fallback rule-based categorization for {len(deltas)} deltas")
        
        result = {
            "high": [],
            "medium": [],
            "low": [],
            "allowed_variance": []
        }
        
        for delta in deltas:
            policy_tag = delta.get('policy', {}).get('tag', '').lower()
            old_val = str(delta.get('old', '')).lower()
            new_val = str(delta.get('new', '')).lower()
            category = delta.get('category', 'unknown')
            
            # Determine bucket using simple rules
            if policy_tag == 'allowed_variance':
                bucket = 'allowed_variance'
            elif any(keyword in old_val or keyword in new_val for keyword in ['password', 'secret', 'key', 'token', 'credential']):
                bucket = 'high'
            elif any(keyword in old_val or keyword in new_val for keyword in ['port', 'host', 'url', 'endpoint', 'jdbc', 'http']):
                bucket = 'medium'
            elif category in ['dependency_added', 'dependency_removed', 'dependency_version_changed']:
                bucket = 'medium'
            elif 'disabled' in new_val or 'false' in new_val:
                bucket = 'medium'
            else:
                bucket = 'low'
            
            # Build item structure - EXACT FORMAT with old/new fields
            item = {
                "id": delta.get('id'),
                "file": file,
                "locator": delta.get('locator', {})
            }
            
            # Add old/new values (exact text that changed)
            item["old"] = str(delta.get('old', '')) if delta.get('old') is not None else None
            item["new"] = str(delta.get('new', '')) if delta.get('new') is not None else None
            
            if bucket == 'allowed_variance':
                # Allowed variance: id, file, locator, old, new, rationale
                item["rationale"] = f"Environment-specific {category} (allowed by policy: {policy_tag})"
            else:
                # High/medium/low: id, file, locator, old, new, why, remediation
                old_str = str(delta.get('old', ''))[:50]
                new_str = str(delta.get('new', ''))[:50]
                
                if 'password' in old_val or 'secret' in old_val:
                    item["why"] = f"Security credential changed from {old_str} to {new_str}"
                elif 'jdbc' in old_val or 'datasource' in old_val:
                    item["why"] = f"Database connection changed from {old_str} to {new_str}"
                elif 'url' in old_val or 'endpoint' in old_val:
                    item["why"] = f"Network endpoint changed from {old_str} to {new_str}"
                elif category.startswith('dependency'):
                    item["why"] = f"Dependency modified: {category}"
                else:
                    item["why"] = f"Configuration changed from {old_str} to {new_str}"
                
                # Generate remediation snippet
                locator_val = delta.get('locator', {}).get('value', 'config.key')
                old_val_snippet = delta.get('old', 'original_value')
                item["remediation"] = {
                    "snippet": f"{locator_val}: {old_val_snippet}"
                }
            
            result[bucket].append(item)
        
        logger.info(f"     ✅ Fallback categorization: High={len(result['high'])}, "
                   f"Medium={len(result['medium'])}, Low={len(result['low'])}, Allowed={len(result['allowed_variance'])}")
        
        return result

    def _merge_llm_outputs(self, llm_outputs: list, overview: dict, context_bundle: dict) -> dict:
        """
        Merge LLM outputs from multiple files into single LLM output with summary statistics.
        
        Args:
            llm_outputs: List of per-file/batch LLM outputs
            overview: Context bundle overview
            context_bundle: Full context bundle for metadata
        
        Returns:
            Single merged LLM output with summary statistics
        """
        logger.info(f"\n📦 Merging {len(llm_outputs)} LLM outputs...")
        
        # Initialize merged structure - EXACT FORMAT (no meta/overview at top level)
        merged = {
            "high": [],
            "medium": [],
            "low": [],
            "allowed_variance": []
        }
        
        # Merge all buckets from all files
        for output in llm_outputs:
            merged["high"].extend(output.get("high", []))
            merged["medium"].extend(output.get("medium", []))
            merged["low"].extend(output.get("low", []))
            merged["allowed_variance"].extend(output.get("allowed_variance", []))
        
        # Sort items within each bucket (by file, then by id)
        def sort_key(item):
            return (item.get("file", ""), item.get("id", ""))
        
        merged["high"] = sorted(merged["high"], key=sort_key)
        merged["medium"] = sorted(merged["medium"], key=sort_key)
        merged["low"] = sorted(merged["low"], key=sort_key)
        merged["allowed_variance"] = sorted(merged["allowed_variance"], key=sort_key)
        
        # Calculate summary statistics from context bundle
        deltas = context_bundle.get("deltas", [])
        file_changes = context_bundle.get("file_changes", {})
        
        # Count files with drift (unique files that have changes)
        files_with_drift = len(set(delta.get("file", "") for delta in deltas))
        
        # Total config files compared
        total_config_files = overview.get("files_compared", 0)
        
        # Total drifts (all deltas)
        total_drifts = len(deltas)
        
        # Add summary statistics to the merged output
        merged["summary"] = {
            "total_config_files": total_config_files,
            "files_with_drift": files_with_drift,
            "total_drifts": total_drifts,
            "high_risk": len(merged["high"]),
            "medium_risk": len(merged["medium"]),
            "low_risk": len(merged["low"]),
            "allowed_variance": len(merged["allowed_variance"])
        }
        
        logger.info(f"✅ Merged LLM output with summary:")
        logger.info(f"   📊 Summary: {total_config_files} config files, {files_with_drift} with drift, {total_drifts} total drifts")
        logger.info(f"   🎯 Risk Distribution: High={len(merged['high'])}, Medium={len(merged['medium'])}, Low={len(merged['low'])}, Allowed={len(merged['allowed_variance'])}")
        
        return merged

    def _parse_ai_json_response(self, ai_response):
        """Robust JSON parsing with multiple fallback strategies"""
        if not ai_response.strip():
            raise json.JSONDecodeError("Empty AI response", ai_response, 0)
        
        # Strategy 1: Try to find complete JSON block
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = ai_response[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"     ⚠️ JSON parsing failed (strategy 1): {e}")
        
        # Strategy 2: Try to fix common JSON issues
        try:
            # Remove any text before first { and after last }
            cleaned = ai_response[ai_response.find('{'):ai_response.rfind('}')+1]
            
            # Fix common issues
            cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
            cleaned = cleaned.replace('},}', '}}')  # Fix double commas
            cleaned = cleaned.replace(',}', '}')    # Fix trailing commas
            cleaned = cleaned.replace(',]', ']')    # Fix trailing commas in arrays
            
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"     ⚠️ JSON parsing failed (strategy 2): {e}")
        
        # Strategy 3: Try to extract partial JSON
        try:
            # Look for the main structure we expect
            if '"delta_analyses"' in ai_response:
                # Try to extract just the delta_analyses array
                start_marker = ai_response.find('"delta_analyses":')
                if start_marker >= 0:
                    # Find the opening bracket
                    bracket_start = ai_response.find('[', start_marker)
                    if bracket_start >= 0:
                        # Count brackets to find the end
                        bracket_count = 0
                        for i in range(bracket_start, len(ai_response)):
                            if ai_response[i] == '[':
                                bracket_count += 1
                            elif ai_response[i] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found the end
                                    partial_json = ai_response[start_idx:ai_response.rfind('}')+1]
                                    # Replace the delta_analyses content with what we found
                                    partial_json = partial_json.replace(
                                        ai_response[start_marker:i+1],
                                        f'"delta_analyses": {ai_response[bracket_start:i+1]}'
                                    )
                                    result = json.loads(partial_json)
                                    return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"     ⚠️ JSON parsing failed (strategy 3): {e}")
        
        # Strategy 4: Create minimal valid JSON
        logger.warning(f"     ⚠️ All JSON parsing strategies failed, creating minimal response")
        return {
            "file_risk_level": "medium",
            "file_verdict": "DRIFT_WARN",
            "delta_analyses": []
        }

    @tool
    async def analyze_configuration_drift(self, 
                                         diff_content: str, 
                                         file_path: str, 
                                         config_type: str = "unknown",
                                         environment: str = "production",
                                         change_type: str = "modification",
                                         additions: int = 0,
                                         deletions: int = 0) -> Dict[str, Any]:
        """
        Analyze configuration drift using AI to provide comprehensive insights.
        
        Args:
            diff_content: The configuration diff content to analyze
            file_path: Path to the configuration file
            config_type: Type of configuration file (e.g., yaml, json, xml, properties)
            environment: Environment context (production, staging, development)
            change_type: Type of change (addition, deletion, modification)
            additions: Number of lines added
            deletions: Number of lines deleted
            
        Returns:
            Analysis results including risk assessment, policy violations, and recommendations
        """
        try:
            prompt = f"""You are an AI-powered configuration drift analysis assistant for the Golden Config AI system. Your task is to analyze configuration changes and provide actionable insights.

### Context:
- **Environment**: {environment}
- **File**: {file_path}
- **Configuration Type**: {config_type}
- **Change Summary**:
  - Lines added: {additions}
  - Lines deleted: {deletions}
  - Change type: {change_type}

### File Importance:
This file is part of the system's configuration. Changes to this file may impact:
- **Security** (e.g., authentication, encryption, access control)
- **Performance** (e.g., resource limits, timeouts, caching)
- **Compliance** (e.g., regulatory requirements, internal policies)
- **Operational stability** (e.g., availability, reliability, maintainability)

### Diff Content:
{diff_content[:2000]}

### Your Task:
Analyze the configuration drift and provide the following in **JSON format**:
1. **risk_level**: Assess the risk level of the changes (low, medium, high, critical) based on their potential impact on security, compliance, performance, and operations.
2. **change_impact**: Describe the functionality or behavior affected by the changes.
3. **policy_violations**: Identify any violations of security, compliance, or operational policies. Reference specific rules where applicable.
4. **recommendations**: Provide actionable steps to address the identified risks or violations. Include preventive measures to avoid similar issues in the future.

### Example Output:
```json
{{
  "risk_level": "high",
  "change_impact": "The change modifies authentication settings, potentially weakening security.",
    "policy_violations": [
        {{
      "type": "security",
      "severity": "high",
      "description": "Authentication must use multi-factor authentication (MFA).",
      "rule": "AUTH_MFA_REQUIRED"
    }}
  ],
    "recommendations": [
        {{
      "priority": "immediate",
      "action": "Revert the change to authentication settings.",
      "rationale": "Current changes disable MFA protection which is required for production systems."
    }},
    {{
      "priority": "high",
      "action": "Ensure MFA is enforced for all users.",
      "rationale": "Multi-factor authentication is a critical security control."
    }},
    {{
      "priority": "medium",
      "action": "Conduct a security review of the configuration.",
      "rationale": "Prevent similar security issues in the future."
    }}
  ],
  "security_impact": "High - Removes critical authentication controls",
  "operational_impact": "Medium - May require user re-authentication"
}}
```

Focus on:
- Security implications (authentication, encryption, access control)
- Operational stability (resource limits, timeouts, availability)
- Compliance requirements (data protection, audit logging)
- Performance impact (throughput, latency, resource usage)
- Best practices adherence (industry standards, organizational policies)"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=1000):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            # Parse JSON response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    analysis["file_path"] = file_path
                    analysis["config_type"] = config_type
                    return analysis
            except json.JSONDecodeError:
                pass
                
            # Fallback if JSON parsing fails
            return {
                "file_path": file_path,
                "config_type": config_type,
                "risk_level": "medium",
                "risk_factors": ["Unable to parse AI response"],
                "policy_violations": [],
                "security_impact": "Requires manual review",
                "operational_impact": "Requires manual review",
                "recommendations": [{"priority": "high", "action": "Manual review required", "rationale": "AI analysis failed"}],
                "change_classification": {"category": "configuration", "severity": "medium", "business_impact": "Unknown"}
            }
            
        except Exception as e:
            logger.error(f"Configuration drift analysis failed: {e}")
            return {
                "error": str(e),
                "file_path": file_path,
                "config_type": config_type,
                "risk_level": "high",
                "recommendations": [{"priority": "immediate", "action": "Manual review required due to analysis error", "rationale": str(e)}]
            }

    @tool
    async def assess_risk_level(self, diff_content: str, file_path: str) -> Dict[str, Any]:
        """
        Assess the risk level of configuration changes.
        
        Args:
            diff_content: The configuration diff content
            file_path: Path to the configuration file
            
        Returns:
            Risk assessment with level and detailed factors
        """
        try:
            prompt = f"""Assess the risk level of the following configuration changes.

File: {file_path}
Changes:
{diff_content[:1500]}

Provide risk assessment in JSON format:
{{
    "risk_level": "low|medium|high|critical",
    "risk_score": 1-100,
    "primary_concerns": ["list of main risk factors"],
    "impact_areas": {{
        "security": "low|medium|high|critical",
        "availability": "low|medium|high|critical", 
        "performance": "low|medium|high|critical",
        "compliance": "low|medium|high|critical"
    }},
    "mitigation_urgency": "immediate|within_24h|within_week|routine"
}}

Consider:
- Critical system components
- Security-sensitive configurations
- Production environment impact
- Regulatory compliance requirements"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=500):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            # Parse JSON response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
                
            return {
                "risk_level": "medium",
                "risk_score": 50,
                "primary_concerns": ["Analysis parsing failed"],
                "impact_areas": {"security": "medium", "availability": "medium", "performance": "low", "compliance": "medium"},
                "mitigation_urgency": "within_24h"
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                "error": str(e),
                "risk_level": "high",
                "risk_score": 80,
                "primary_concerns": [f"Assessment error: {str(e)}"],
                "mitigation_urgency": "immediate"
            }

    @tool
    async def check_policy_violations(self, diff_content: str, file_path: str) -> Dict[str, Any]:
        """
        Check for policy violations in configuration changes.
        
        Args:
            diff_content: The configuration diff content
            file_path: Path to the configuration file
            
        Returns:
            Policy violation assessment
        """
        try:
            prompt = f"""Check the following configuration changes for policy violations.

File: {file_path}
Changes:
{diff_content[:1500]}

Check against common policies and provide results in JSON:
{{
    "violations_found": true/false,
    "violation_count": number,
    "violations": [
        {{
            "policy_type": "security|compliance|operational|performance",
            "severity": "low|medium|high|critical",
            "description": "what policy was violated",
            "specific_issue": "exact configuration causing violation",
            "remediation": "how to fix the violation"
        }}
    ],
    "compliance_score": 0-100,
    "policy_categories_checked": ["security", "compliance", "operational"]
}}

Check for violations of:
- Security policies (encryption, authentication, access control)
- Compliance requirements (logging, data retention, audit trails)
- Operational standards (resource limits, monitoring, backup)
- Performance guidelines (timeouts, connection limits)"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=700):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            # Parse JSON response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
                
            return {
                "violations_found": False,
                "violation_count": 0,
                "violations": [],
                "compliance_score": 100,
                "policy_categories_checked": ["security", "compliance", "operational"],
                "note": "Policy check completed but response parsing failed"
            }
            
        except Exception as e:
            logger.error(f"Policy violation check failed: {e}")
            return {
                "error": str(e),
                "violations_found": True,
                "violation_count": 1,
                "violations": [{"policy_type": "operational", "severity": "high", "description": f"Policy check failed: {str(e)}"}]
            }

    @tool
    async def assess_overall_drift_risk(self, 
                                       total_files_changed: int,
                                       risk_distribution: Dict[str, int],
                                       files_with_violations: int,
                                       environment: str = "production") -> Dict[str, Any]:
        """
        Use AI to assess overall drift risk comprehensively across all configuration changes.
        
        Args:
            total_files_changed: Total number of configuration files changed
            risk_distribution: Distribution of risk levels (e.g., {"high": 3, "medium": 5, "low": 2})
            files_with_violations: Number of files with policy violations
            environment: Environment context (production, staging, development)
            
        Returns:
            AI-powered overall risk assessment with mitigation strategies
        """
        logger.info(f"🤖 AI-powered overall risk assessment for {environment} environment")
        
        try:
            prompt = f"""You are an AI-powered configuration drift analysis assistant for the Golden Config AI system. Your task is to assess the overall risk of configuration changes across multiple files and provide actionable insights.

### Context:
- **Environment**: {environment}
- **Drift Summary**:
  - Total files changed: {total_files_changed}
  - Risk level distribution: {risk_distribution}
  - Files with policy violations: {files_with_violations}

### Key Considerations:
1. **Environment Criticality**:
   - Consider the importance of the environment (e.g., production vs staging) and its tolerance for risk.
   - Production environments have lower tolerance for risk and require immediate attention.
   - Staging/development environments can tolerate more risk but still require review.

2. **Change Volume and Complexity**:
   - Assess the overall volume of changes and their complexity.
   - Multiple high-risk changes increase overall risk exponentially.
   - Large number of medium-risk changes can be as concerning as a few high-risk changes.

3. **Security Implications**:
   - Identify any potential security risks introduced by the changes.
   - Security issues in production require immediate attention.
   - Multiple security violations indicate systemic issues.

4. **Business Impact**:
   - Evaluate the potential impact of the changes on business operations, compliance, and performance.
   - Consider cascading failures and dependencies.
   - Assess potential for service disruption.

### Your Task:
Analyze the overall configuration drift and provide the following in **JSON format**:
1. **overall_risk_level**: Assess the overall risk level (low, medium, high, critical) based on the drift summary and key considerations.
2. **risk_factors**: List the key factors contributing to the overall risk level. Be specific about numbers and context.
3. **mitigation_strategies**: Provide actionable strategies to mitigate the identified risks. Prioritize by urgency.

### Example Output:
```json
{{
  "overall_risk_level": "high",
  "risk_factors": [
    "5 high-risk changes detected in production environment",
    "3 files with security policy violations",
    "Critical environment: production - zero tolerance for security issues",
    "Large change volume ({total_files_changed} files) increases deployment risk"
  ],
  "mitigation_strategies": [
    "Conduct immediate detailed review of all high-risk changes before any deployment",
    "Resolve all security policy violations - these are blocking issues for production",
    "Perform comprehensive security testing in staging environment first",
    "Implement phased rollout strategy to minimize blast radius",
    "Prepare rollback plan for each changed configuration file",
    "Schedule change review meeting with security and operations teams"
  ],
  "mitigation_priority": "urgent"
}}
```

### Risk Level Guidelines:
- **Critical**: Production + (high-risk changes > 3 OR security violations > 2)
- **High**: Production + (high-risk changes > 0 OR security violations > 0) OR Non-prod + (critical changes > 0)
- **Medium**: Multiple medium-risk changes OR few high-risk in non-prod
- **Low**: Few low/medium-risk changes in non-critical environments

Provide your assessment:"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=800):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            # Parse JSON response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    ai_assessment = json.loads(json_str)
                    return ai_assessment
            except json.JSONDecodeError:
                pass
                
            # Fallback assessment based on simple rules
            high_risk_count = risk_distribution.get('high', 0) + risk_distribution.get('critical', 0)
            
            if environment == 'production' and (high_risk_count > 3 or files_with_violations > 2):
                overall_risk = 'critical'
                priority = 'urgent'
            elif environment == 'production' and (high_risk_count > 0 or files_with_violations > 0):
                overall_risk = 'high'
                priority = 'urgent'
            elif high_risk_count > 5:
                overall_risk = 'high'
                priority = 'high'
            elif total_files_changed > 10:
                overall_risk = 'medium'
                priority = 'standard'
            else:
                overall_risk = 'low'
                priority = 'routine'
            
            return {
                "overall_risk_level": overall_risk,
                "risk_factors": [
                    f"{high_risk_count} high/critical-risk changes detected",
                    f"{files_with_violations} files with policy violations",
                    f"Environment: {environment}"
                ],
                "mitigation_strategies": [
                    "Review all high-risk changes before deployment",
                    "Resolve policy violations",
                    "Test changes in staging environment"
                ],
                "mitigation_priority": priority
            }
            
        except Exception as e:
            logger.error(f"Overall risk assessment failed: {e}")
            # Safe fallback
            return {
                "overall_risk_level": "medium",
                "risk_factors": [f"Assessment error: {str(e)}"],
                "mitigation_strategies": ["Manual review required"],
                "mitigation_priority": "standard"
            }

    @tool
    async def generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            analysis_results: Combined results from previous analysis tools
            
        Returns:
            Structured recommendations for addressing identified issues
        """
        try:
            prompt = f"""Based on the following configuration analysis results, generate actionable recommendations.

Analysis Results:
{json.dumps(analysis_results, indent=2)[:2000]}

Provide recommendations in JSON format:
{{
    "immediate_actions": [
        {{
            "action": "specific action to take immediately",
            "priority": "critical|high|medium|low",
            "estimated_effort": "time estimate",
            "risk_if_delayed": "consequence of not acting"
        }}
    ],
    "short_term_actions": ["actions for next 1-7 days"],
    "long_term_improvements": ["strategic improvements for future"],
    "monitoring_recommendations": ["what to monitor going forward"],
    "preventive_measures": ["how to prevent similar issues"],
    "approval_required": true/false,
    "rollback_plan": "recommended rollback approach if needed",
    "testing_recommendations": ["specific tests to perform"]
}}

Consider:
- Risk level and urgency
- Business impact
- Resource requirements
- Dependencies and prerequisites"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            ai_response = ""
            async for event in self.model.stream(messages, max_tokens=800):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        ai_response += delta["text"]
            
            # Parse JSON response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
                
            return {
                "immediate_actions": [{"action": "Manual review required", "priority": "high", "estimated_effort": "30 minutes"}],
                "short_term_actions": ["Review configuration changes"],
                "long_term_improvements": ["Implement automated policy checking"],
                "monitoring_recommendations": ["Monitor configuration drift"],
                "preventive_measures": ["Regular configuration audits"],
                "approval_required": True,
                "rollback_plan": "Revert to previous configuration version",
                "testing_recommendations": ["Test in staging environment"]
            }
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                "error": str(e),
                "immediate_actions": [{"action": f"Address recommendation generation error: {str(e)}", "priority": "high"}]
            }
