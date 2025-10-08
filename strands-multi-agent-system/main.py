#!/usr/bin/env python3
"""
Golden Config AI - Multi-Agent System Main Server

This server orchestrates the complete multi-agent validation workflow:
- Supervisor Agent coordinates the pipeline
- Config Collector Agent fetches Git diffs
- Diff Policy Engine Agent analyzes with AI

Runs on localhost:3000 for easy access.
"""

import uvicorn
import json
import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

# Strands agent system imports
from shared.config import Config
from Agents.Supervisor.supervisor_agent import run_validation


# Setup templates directory
templates_dir = Path(__file__).parent / "api" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize FastAPI app
app = FastAPI(
    title="Golden Config AI - Multi-Agent System",
    description="Complete Configuration Drift Analysis with Supervisor + Worker Agents",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load default values from environment
config = Config()
DEFAULT_REPO_URL = os.getenv("DEFAULT_REPO_URL", "https://gitlab.verizon.com/saja9l7/golden_config.git")
DEFAULT_GOLDEN_BRANCH = os.getenv("DEFAULT_GOLDEN_BRANCH", "gold")
DEFAULT_DRIFT_BRANCH = os.getenv("DEFAULT_DRIFT_BRANCH", "drift")

# Debug: Print loaded environment variables
print(f"🔧 Environment Variables Loaded:")
print(f"   DEFAULT_REPO_URL: {DEFAULT_REPO_URL}")
print(f"   DEFAULT_GOLDEN_BRANCH: {DEFAULT_GOLDEN_BRANCH}")
print(f"   DEFAULT_DRIFT_BRANCH: {DEFAULT_DRIFT_BRANCH}")
print()

# Request models
class ValidationRequest(BaseModel):
    """Request for configuration drift validation"""
    repo_url: str = Field(
        default=DEFAULT_REPO_URL,
        description="GitLab repository URL"
    )
    golden_branch: str = Field(
        default=DEFAULT_GOLDEN_BRANCH,
        description="Golden/reference branch name"
    )
    drift_branch: str = Field(
        default=DEFAULT_DRIFT_BRANCH,
        description="Drift/comparison branch name"
    )
    target_folder: str = Field(
        default="",
        description="Optional: specific folder to analyze (empty = entire repo)"
    )
    project_id: str = Field(
        default="config-validation",
        description="Project identifier"
    )
    mr_iid: str = Field(
        default="auto",
        description="Merge request ID or validation identifier"
    )
    
    @field_validator('repo_url')
    @classmethod
    def validate_repo_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('repo_url must be a valid HTTP/HTTPS URL')
        return v


class QuickAnalysisRequest(BaseModel):
    """Quick analysis with default settings"""
    pass


# Global state
latest_results: Optional[Dict[str, Any]] = None
validation_in_progress: bool = False


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the main UI dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "service": "Golden Config AI - Multi-Agent System",
        "version": "2.0.0",
        "status": "running",
        "architecture": "supervisor_orchestration",
        "agents": {
            "supervisor": "Orchestrates the validation workflow",
            "config_collector": "Fetches Git diffs and analyzes changes",
            "diff_policy_engine": "AI-powered drift analysis and policy validation"
        },
        "communication": "file_based",
        "default_repo": DEFAULT_REPO_URL,
        "default_branches": {
            "golden": DEFAULT_GOLDEN_BRANCH,
            "drift": DEFAULT_DRIFT_BRANCH
        },
        "endpoints": {
            "ui": "GET /",
            "validate": "POST /api/validate",
            "quick_analyze": "POST /api/analyze/quick",
            "latest_results": "GET /api/latest-results",
            "validation_status": "GET /api/validation-status",
            "config": "GET /api/config",
            "llm_output": "GET /api/llm-output",
            "health": "GET /health"
        }
    }


@app.get("/api/validation-status")
async def validation_status():
    """Check if validation is in progress"""
    global validation_in_progress
    
    return {
        "in_progress": validation_in_progress,
        "has_results": latest_results is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/validate")
async def validate_configuration(request: ValidationRequest, background_tasks: BackgroundTasks):
    """
    Run complete multi-agent validation workflow.
    
    This orchestrates:
    1. Supervisor Agent - Creates validation run and coordinates workflow
    2. Config Collector Agent - Fetches Git diffs from repository
    3. Diff Policy Engine Agent - AI-powered drift analysis
    
    Returns file paths to analysis results.
    """
    global validation_in_progress, latest_results
    
    if validation_in_progress:
        raise HTTPException(
            status_code=409,
            detail="Validation already in progress. Please wait for completion."
        )
    
    print("=" * 80)
    print("🚀 MULTI-AGENT VALIDATION REQUEST")
    print("=" * 80)
    print(f"📦 Repository: {request.repo_url}")
    print(f"🌿 Golden Branch: {request.golden_branch}")
    print(f"🔀 Drift Branch: {request.drift_branch}")
    print(f"📁 Target Folder: {request.target_folder or 'entire repository'}")
    print(f"🆔 Project ID: {request.project_id}")
    print(f"🔢 MR/ID: {request.mr_iid}")
    print("=" * 80)
    
    try:
        validation_in_progress = True
        start_time = datetime.now()
        
        # Generate MR ID if auto
        mr_iid = request.mr_iid
        if mr_iid == "auto":
            mr_iid = f"val_{int(datetime.now().timestamp())}"
        
        print("\n🤖 Starting Supervisor Agent orchestration...")
        print("   ├─ Supervisor Agent: Coordinates workflow")
        print("   ├─ Config Collector Agent: Fetches Git diffs")
        print("   └─ Diff Policy Engine Agent: AI-powered analysis")
        print()
        
        # Run validation through supervisor
        result = run_validation(
            project_id=request.project_id,
            mr_iid=mr_iid,
            repo_url=request.repo_url,
            golden_branch=request.golden_branch,
            drift_branch=request.drift_branch,
            target_folder=request.target_folder
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("✅ VALIDATION COMPLETED")
        print("=" * 80)
        print(f"⏱️  Execution Time: {execution_time:.2f}s")
        print(f"🆔 Run ID: {result.get('run_id', 'N/A')}")
        print(f"📊 Verdict: {result.get('verdict', 'N/A')}")
        print("=" * 80)
        
        # Try to load enhanced analysis data if available
        enhanced_data = None
        try:
            # Look for the enhanced analysis file in the result
            if "data" in result and "file_paths" in result["data"]:
                enhanced_file = result["data"]["file_paths"].get("enhanced_analysis")
                if enhanced_file and Path(enhanced_file).exists():
                    with open(enhanced_file, 'r', encoding='utf-8') as f:
                        enhanced_data = json.load(f)
                    print(f"✅ Loaded enhanced analysis data from: {enhanced_file}")
        except Exception as e:
            print(f"⚠️ Could not load enhanced analysis data: {e}")
        
        # Prepare response with enhanced data if available
        validation_result = result
        if enhanced_data:
            # Merge enhanced data into validation result
            validation_result = {
                **result,
                "enhanced_data": enhanced_data,
                "clusters": enhanced_data.get("clusters", []),
                "analyzed_deltas": enhanced_data.get("analyzed_deltas_with_ai", []),
                "total_clusters": len(enhanced_data.get("clusters", [])),
                "policy_violations": enhanced_data.get("policy_violations", []),
                "policy_violations_count": len(enhanced_data.get("policy_violations", [])),
                "overall_risk_level": enhanced_data.get("overall_risk_level", "unknown"),
                "verdict": enhanced_data.get("verdict", "UNKNOWN"),
                "environment": enhanced_data.get("environment", "unknown"),
                "critical_violations": len([v for v in enhanced_data.get("policy_violations", []) if v.get('severity') == 'critical']),
                "high_violations": len([v for v in enhanced_data.get("policy_violations", []) if v.get('severity') == 'high'])
            }
        
        response = {
            "status": "success",
            "architecture": "multi_agent_supervisor",
            "agents_used": ["supervisor", "config_collector", "diff_policy_engine"],
            "communication_method": "file_based",
            "validation_result": validation_result,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_params": {
                "repo_url": request.repo_url,
                "golden_branch": request.golden_branch,
                "drift_branch": request.drift_branch,
                "target_folder": request.target_folder or "/"
            }
        }
        
        latest_results = response
        
        return response
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
    
    finally:
        validation_in_progress = False


@app.post("/api/analyze/quick")
async def quick_analyze(request: QuickAnalysisRequest):
    """
    Quick analysis with default settings from environment variables.
    
    This is a convenience endpoint that uses predefined repository and branches from .env
    """
    print("🚀 Quick Analysis Request (using defaults from .env)")
    
    default_request = ValidationRequest(
        repo_url=DEFAULT_REPO_URL,
        golden_branch=DEFAULT_GOLDEN_BRANCH,
        drift_branch=DEFAULT_DRIFT_BRANCH,
        target_folder="",
        project_id="quick_analysis",
        mr_iid="quick_analysis"
    )
    
    # Use background tasks to avoid timeout
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    
    return await validate_configuration(default_request, background_tasks)


@app.get("/api/latest-results")
async def get_latest_results():
    """Get the latest validation results"""
    if latest_results:
        return latest_results
    else:
        raise HTTPException(status_code=404, detail="No validation results available yet")


@app.get("/api/sample-data")
async def get_sample_data():
    """
    Trigger a quick analysis for sample data.
    This is for UI compatibility with the old agent_analysis_server.
    """
    return await quick_analyze(QuickAnalysisRequest())


@app.post("/api/analyze/agent")
async def analyze_agent_compat(request: Dict[str, Any]):
    """
    Compatibility endpoint for UI that expects /api/analyze/agent.
    Maps to the new validation endpoint.
    """
    print("🔄 Legacy endpoint called (/api/analyze/agent), redirecting to new validation...")
    
    validation_request = ValidationRequest(
        repo_url=request.get("repo_url", DEFAULT_REPO_URL),
        golden_branch=request.get("golden_branch", DEFAULT_GOLDEN_BRANCH),
        drift_branch=request.get("drift_branch", DEFAULT_DRIFT_BRANCH),
        target_folder=request.get("target_folder", ""),
        project_id=request.get("project_id", "config-validation"),
        mr_iid=request.get("mr_iid", "auto")
    )
    
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    
    return await validate_configuration(validation_request, background_tasks)


@app.get("/api/agent-status")
async def agent_status():
    """Check agent system status"""
    try:
        config = Config()
        return {
            "status": "initialized",
            "architecture": "multi_agent_supervisor",
            "agents": {
                "supervisor": {
                    "status": "ready",
                    "description": "Orchestrates validation workflow",
                    "model": config.bedrock_model_id
                },
                "config_collector": {
                    "status": "ready",
                    "description": "Fetches Git diffs",
                    "model": config.bedrock_worker_model_id
                },
                "diff_policy_engine": {
                    "status": "ready",
                    "description": "AI-powered drift analysis",
                    "model": config.bedrock_worker_model_id
                }
            },
            "communication": "file_based",
            "output_location": "config_data/",
            "message": "All agents ready for validation"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Agent initialization failed"
        }


@app.get("/api/config")
async def get_config():
    """Get environment configuration for UI"""
    return {
        "repo_url": DEFAULT_REPO_URL,
        "golden_branch": DEFAULT_GOLDEN_BRANCH,
        "drift_branch": DEFAULT_DRIFT_BRANCH,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/llm-output")
async def get_llm_output():
    """Get the latest LLM output in adjudicator format"""
    import glob
    
    llm_output_files = sorted(glob.glob("config_data/llm_output/llm_output_*.json"), reverse=True)
    
    if llm_output_files:
        try:
            with open(llm_output_files[0], 'r', encoding='utf-8') as f:
                llm_output = json.load(f)
            
            return {
                "status": "success",
                "file_path": llm_output_files[0],
                "data": llm_output
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load LLM output: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="No LLM output files available yet")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        "status": "healthy",
        "service": "Golden Config AI - Multi-Agent System",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "architecture": "supervisor_orchestration",
        "validation_in_progress": validation_in_progress,
        "has_results": latest_results is not None
    }


def main():
    """Start the multi-agent validation server"""
    print("\n" + "=" * 80)
    print("🚀 GOLDEN CONFIG AI - MULTI-AGENT SYSTEM")
    print("=" * 80)
    print()
    print("🌐 Server URLs:")
    print(f"   Dashboard:  http://localhost:3000")
    print(f"   API Docs:   http://localhost:3000/docs")
    print(f"   Health:     http://localhost:3000/health")
    print()
    print("🤖 AGENT ARCHITECTURE:")
    print("   ┌─────────────────────────────────┐")
    print("   │     Supervisor Agent            │  ← Orchestrates workflow")
    print("   │  (Claude 3.5 Sonnet)            │")
    print("   └──────────┬──────────────────────┘")
    print("              │")
    print("              ├──► Config Collector Agent")
    print("              │    (Fetches Git diffs)")
    print("              │    (Claude 3 Haiku)")
    print("              │")
    print("              └──► Diff Policy Engine Agent")
    print("                   (AI-powered analysis)")
    print("                   (Claude 3 Haiku)")
    print()
    print("💾 Communication: File-Based")
    print("   ├─ Config Collector → config_data/drift_analysis/*.json")
    print("   ├─ Diff Engine     → config_data/diff_analysis/*.json")
    print("   └─ Supervisor      → config_data/reports/*.md")
    print()
    print("🎯 DEFAULT CONFIGURATION (from .env):")
    print(f"   Repository: {DEFAULT_REPO_URL}")
    print(f"   Golden Branch: {DEFAULT_GOLDEN_BRANCH}")
    print(f"   Drift Branch:  {DEFAULT_DRIFT_BRANCH}")
    print()
    print("📚 ENDPOINTS:")
    print("   POST /api/validate          - Run full validation (custom params)")
    print("   POST /api/analyze/quick     - Quick analysis (default settings)")
    print("   POST /api/analyze/agent     - Legacy compatibility endpoint")
    print("   GET  /api/latest-results    - Get most recent validation results")
    print("   GET  /api/validation-status - Check if validation is running")
    print("   GET  /api/agent-status      - Check agent system status")
    print()
    print("🎮 USAGE:")
    print("   1. Open http://localhost:3000 in your browser")
    print("   2. Click 'Load Sample Data' or 'Analyze' to start validation")
    print("   3. Watch the multi-agent system coordinate the analysis")
    print("   4. Review comprehensive drift analysis results")
    print()
    print("✨ FEATURES:")
    print("   ✅ Complete Supervisor orchestration")
    print("   ✅ File-based inter-agent communication")
    print("   ✅ Real GitLab repository analysis")
    print("   ✅ AI-powered drift detection with enhanced prompts")
    print("   ✅ Comprehensive risk assessment")
    print("   ✅ Policy violation detection")
    print("   ✅ Actionable recommendations")
    print()
    print("🛑 Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=3000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

