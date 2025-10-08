"""
Configuration Diff Collection Agent (Strands Implementation)

This agent collects configuration diffs between golden and drift branches,
identifies configuration files, and produces structured diff reports for
use by downstream Golden Config AI agents. Functionality, paths, inputs
and outputs are preserved from the original procedural script.
"""

from datetime import datetime
import asyncio
import logging
import os
import sys
import json
import argparse
import tempfile
import shutil
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import git
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools import tool
from dotenv import load_dotenv
from shared.models import TaskResponse

# Load environment variables (maintains original behavior)
load_dotenv()

# Import drift.py analysis functions for precision analysis
from shared.drift_analyzer import (
    extract_repo_tree,
    classify_files,
    diff_structural,
    semantic_config_diff,
    extract_dependencies,
    dependency_diff,
    detector_spring_profiles,
    detector_jenkinsfile,
    build_code_hunk_deltas,
    build_binary_deltas,
    emit_context_bundle,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Original helper functions (preserved exactly in logic / naming)
# ---------------------------------------------------------------------------

def setup_git_auth(repo_url: str) -> str:
    """Set up Git authentication using environment variables (unchanged)."""
    gitlab_token = os.getenv('GITLAB_TOKEN')
    gitlab_username = os.getenv('GITLAB_USERNAME')
    gitlab_password = os.getenv('GITLAB_PASSWORD')
    if gitlab_token:
        print("✅ Using GitLab personal access token for authentication.")
        if repo_url.startswith('https://'):
            return repo_url.replace('https://', f'https://oauth2:{gitlab_token}@')
    elif gitlab_username and gitlab_password:
        print("✅ Using username/password for authentication.")
        if repo_url.startswith('https://'):
            return repo_url.replace('https://', f'https://{gitlab_username}:{gitlab_password}@')
    print("⚠️ No authentication credentials found in environment variables. Proceeding without auth.")
    return repo_url

def configure_git_user():
    """Configures Git user settings from environment variables."""
    git_user_name = os.getenv('GIT_USER_NAME')
    git_user_email = os.getenv('GIT_USER_EMAIL')
    
    if git_user_name and git_user_email:
        try:
            os.system(f'git config --global user.name "{git_user_name}"')
            os.system(f'git config --global user.email "{git_user_email}"')
            print(f"✅ Git user configured as: {git_user_name} <{git_user_email}>")
        except Exception as e:
            print(f"❌ Warning - Could not configure git user: {e}")

def ensure_repo_ready(repo_url: str, repo_path: Path) -> Optional[git.Repo]:
    """Clone or fetch repository in temporary location."""
    try:
        if repo_path.exists():
            print(f"[INFO] Repository already exists at: {repo_path}")
            repo = git.Repo(repo_path)
            print("[INFO] Fetching latest updates from remote...")
            repo.remotes.origin.fetch()
            return repo
        else:
            print(f"[INFO] Cloning repository into temporary location: {repo_path}")
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            authenticated_url = setup_git_auth(repo_url)
            repo = git.Repo.clone_from(authenticated_url, repo_path)
            print("[INFO] Fetching origin after clone...")
            repo.remotes.origin.fetch()
            print("✅ Repository is ready.")
            return repo
    except Exception as e:
        print(f"[ERR] Failed to clone or access repository: {e}")
        return None

def switch_to_branch(repo: git.Repo, branch_name: str) -> Optional[str]:
    """Switch branches (unchanged logic)."""
    try:
        original_branch = repo.active_branch.name
        print(f"ℹ️ Current branch is '{original_branch}'.")
        if original_branch != branch_name:
            print(f"Attempting to switch to branch '{branch_name}'...")
            repo.git.checkout(branch_name)
            print(f"✅ Switched to branch '{branch_name}'.")
        else:
            print(f"ℹ️ Already on branch '{branch_name}'. No switch needed.")
        return original_branch
    except git.exc.GitCommandError as e:
        print(f"❌ Could not checkout branch '{branch_name}': {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error switching branch: {e}")
        return None

def is_config_file(file_path: str) -> bool:
    config_extensions = {'.yml', '.yaml', '.json', '.env', '.ini', '.cfg', '.conf', '.toml', '.xml', '.properties', '.config'}
    config_filenames = {'dockerfile', 'docker-compose', 'makefile', 'requirements.txt', 'package.json', 'package-lock.json', 'poetry.lock', 'pipfile', 'setup.py', 'setup.cfg', 'pyproject.toml', '.gitignore', '.dockerignore', 'webpack.config.js', 'babel.config.js'}
    file_name = Path(file_path).name.lower()
    file_suffix = Path(file_path).suffix.lower()
    return (file_suffix in config_extensions or file_name in config_filenames or (not file_suffix and file_name in {'dockerfile', 'makefile', 'jenkinsfile'}))

def get_change_type(status: str) -> str:
    """Convert git status to human-readable change type."""
    if status.startswith('A'):
        return "added"
    elif status.startswith('D'):
        return "deleted"
    elif status.startswith('M'):
        return "modified"
    elif status.startswith('R'):
        return "renamed"
    elif status.startswith('C'):
        return "copied"
    else:
        return "unknown"

def get_config_file_paths(repo: git.Repo, target_folder: str = None) -> List[str]:

    """
    Get list of configuration files in the repository.
    """
    config_extensions = {
        '.yml', '.yaml', '.json', '.env', '.ini', '.cfg', '.conf',
        '.toml', '.xml', '.properties', '.config'
    }
    
    config_filenames = {
        'dockerfile', 'docker-compose', 'makefile', 'requirements.txt',
        'package.json', 'package-lock.json', 'poetry.lock', 'pipfile',
        'setup.py', 'setup.cfg', 'pyproject.toml', '.gitignore',
        '.dockerignore', 'webpack.config.js', 'babel.config.js'
    }
    
    relative_paths = []
    

    try:
        repo_root = Path(repo.working_tree_dir)
        search_root = repo_root / target_folder if target_folder else repo_root
        
        if not search_root.exists():
            print(f"❌ Target folder '{target_folder}' does not exist in the repository.")
            return []
        
        print(f"🔍 Searching for configuration files in: {search_root}")
        print(f"📁 Repository root: {repo_root}")
        
        # Walk through the directory tree
        for file_path in search_root.rglob("*"):
            if file_path.is_file():
                # Get relative path from repository root
                relative_path = file_path.relative_to(repo_root)
                relative_path_str = str(relative_path).replace("\\", "/")  # Normalize path separators
                
                # Check if it's a config file by extension
                if file_path.suffix.lower() in config_extensions:
                    relative_paths.append(relative_path_str)
                # Check if it's a config file by name
                elif file_path.name.lower() in config_filenames:
                    relative_paths.append(relative_path_str)
                # Special case for files without extensions that might be config files
                elif not file_path.suffix and file_path.name.lower() in {'dockerfile', 'makefile', 'jenkinsfile'}:
                    relative_paths.append(relative_path_str)
        print(f"✅ Found {len(relative_paths)} configuration files")
    except Exception as e:
        print(f"❌ Error scanning repository: {e}")
        return []
    return sorted(relative_paths)

def collect_branch_diffs(repo: git.Repo, golden_branch: str, drift_branch: str, target_folder: str = None) -> Dict[str, Any]:
    print(f"\n🔍 Starting diff collection between '{golden_branch}' and '{drift_branch}'")
    try:
        original_branch = repo.active_branch.name
        print(f"ℹ️ Original branch: {original_branch}")
    except:
        original_branch = None
        print("ℹ️ Could not determine original branch")
    diff_results: Dict[str, Any] = {
        'analysis_metadata': {
            'golden_branch': golden_branch,
            'drift_branch': drift_branch,
            'target_folder': target_folder or 'entire_repository',
            'analysis_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'generated_by': 'config_collector/config_collector_agent.py'
        },
        'branch_comparison': {
            'files_changed': [],
            'files_added_in_drift': [],
            'files_removed_in_drift': [],
            'files_modified': []
        },
        'detailed_diffs': {},
        'summary': {
            'total_files_analyzed': 0,
            'files_with_changes': 0,
            'lines_added': 0,
            'lines_removed': 0,
            'files_added': 0,
            'files_removed': 0
        }
    }
    try:
        print('🔄 Checking branch availability...')
        try:
            repo.git.show_ref(f'refs/heads/{golden_branch}')
            print(f"✅ Golden branch '{golden_branch}' exists")
        except git.exc.GitCommandError:
            try:
                repo.git.show_ref(f'refs/remotes/origin/{golden_branch}')
                print(f"✅ Golden branch 'origin/{golden_branch}' exists")
                golden_branch = f'origin/{golden_branch}'
            except git.exc.GitCommandError:
                print(f"❌ Golden branch '{golden_branch}' not found")
                return diff_results
        try:
            repo.git.show_ref(f'refs/heads/{drift_branch}')
            print(f"✅ Drift branch '{drift_branch}' exists")
        except git.exc.GitCommandError:
            try:
                repo.git.show_ref(f'refs/remotes/origin/{drift_branch}')
                print(f"✅ Drift branch 'origin/{drift_branch}' exists")
                drift_branch = f'origin/{drift_branch}'
            except git.exc.GitCommandError:
                print(f"❌ Drift branch '{drift_branch}' not found")
                return diff_results
        print(f"📊 Generating diff between {golden_branch} and {drift_branch}...")
        diff_index = repo.git.diff(f'{golden_branch}..{drift_branch}', name_status=True)
        if not diff_index:
            print('ℹ️ No differences found between branches')
            return diff_results
        changed_files = []
        for line in diff_index.split('\n'):
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    status, file_path = parts[0], parts[1]
                    if target_folder and not file_path.startswith(target_folder):
                        continue
                    if is_config_file(file_path):
                        changed_files.append((status, file_path))
        print(f"📁 Found {len(changed_files)} changed configuration files")
        for status, file_path in changed_files:
            print(f"🔄 Processing {status}: {file_path}")
            if status.startswith('A'):
                diff_results['branch_comparison']['files_added_in_drift'].append(file_path)
                diff_results['summary']['files_added'] += 1
            elif status.startswith('D'):
                diff_results['branch_comparison']['files_removed_in_drift'].append(file_path)
                diff_results['summary']['files_removed'] += 1
            elif status.startswith('M'):
                diff_results['branch_comparison']['files_modified'].append(file_path)
                try:
                    detailed_diff = repo.git.diff(f'{golden_branch}..{drift_branch}', file_path)
                    diff_results['detailed_diffs'][file_path] = {
                        'status': 'modified',
                        'diff_content': detailed_diff,
                        'lines_added': detailed_diff.count('\n+') - detailed_diff.count('\n+++'),
                        'lines_removed': detailed_diff.count('\n-') - detailed_diff.count('\n---')
                    }
                    diff_results['summary']['lines_added'] += diff_results['detailed_diffs'][file_path]['lines_added']
                    diff_results['summary']['lines_removed'] += diff_results['detailed_diffs'][file_path]['lines_removed']
                except Exception as e:
                    print(f"❌ Could not get detailed diff for {file_path}: {e}")
                    diff_results['detailed_diffs'][file_path] = {
                        'status': 'modified',
                        'diff_content': f'Error getting diff: {e}',
                        'lines_added': 0,
                        'lines_removed': 0
                    }
            diff_results['branch_comparison']['files_changed'].append({'file_path': file_path, 'status': status, 'change_type': get_change_type(status)})
        diff_results['summary']['total_files_analyzed'] = len(changed_files)
        diff_results['summary']['files_with_changes'] = len(changed_files)
        print('✅ Diff collection complete!')
        print(f"   • Files analyzed: {diff_results['summary']['total_files_analyzed']}")
        print(f"   • Files modified: {len(diff_results['branch_comparison']['files_modified'])}")
        print(f"   • Files added: {diff_results['summary']['files_added']}")
        print(f"   • Files removed: {diff_results['summary']['files_removed']}")
    except Exception as e:
        print(f"❌ Error during diff collection: {e}")
        diff_results['error'] = str(e)
    finally:
        if 'original_branch' in locals() and original_branch:
            try:
                print(f"🔄 Restoring original branch: {original_branch}")
                repo.git.checkout(original_branch)
                print(f"✅ Restored to branch: {original_branch}")
            except Exception as e:
                print(f"❌ Warning - Could not restore original branch: {e}")
    return diff_results

# ---------------------------------------------------------------------------
# Agent Class (following diff_engine_agent.py pattern)
# ---------------------------------------------------------------------------
class ConfigCollectorAgent(Agent):
    """Config Collector Agent

    Provides async @tool access to diff collection and repository utilities
    using only serializable types (following diff_engine_agent.py pattern).
    """
    def __init__(self, config: Optional[Any] = None):
        system_prompt = self._get_config_collector_prompt()
        super().__init__(
            model=BedrockModel(model_id=(getattr(config, 'bedrock_model_id', None)) ),
            system_prompt=system_prompt,
            tools=[
                self.collect_repository_diffs,
                self.analyze_config_files,
                self.setup_repository_access,
                self.run_complete_diff_workflow
            ]
        )
        self.config = config
    
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like git.Repo
    
    def process_task(self, task: 'TaskRequest') -> 'TaskResponse':
        """
        Process a task to collect configuration diffs.
        
        This is the main entry point when called from Supervisor or API.
        Runs the complete diff workflow and saves results to file.
        
        Args:
            task: TaskRequest with parameters:
                - repo_url: Repository URL
                - golden_branch: Golden/reference branch
                - drift_branch: Drift/comparison branch
                - target_folder: Optional folder to analyze
                
        Returns:
            TaskResponse with drift_analysis_file path
        """
        start_time = time.time()  # Track processing time
        
        try:
            logger.info(f"🔍 Config Collector processing task: {task.task_id}")
            
            params = task.parameters
            
            # Extract parameters (use environment defaults)
            repo_url = params.get('repo_url')
            golden_branch = params.get('golden_branch', os.getenv('DEFAULT_GOLDEN_BRANCH', 'gold'))
            drift_branch = params.get('drift_branch', os.getenv('DEFAULT_DRIFT_BRANCH', 'drift'))
            target_folder = params.get('target_folder', '')
            
            if not repo_url:
                return TaskResponse(
                    task_id=task.task_id,
                    status="failed",
                    result={},
                    error="Missing required parameter: repo_url",
                    processing_time_seconds=time.time() - start_time,
                    metadata={"agent": "config_collector"}
                )
            
            # Run complete diff workflow (saves to file)
            result = asyncio.run(self.run_complete_diff_workflow(
                repo_url=repo_url,
                golden_branch=golden_branch,
                drift_branch=drift_branch,
                target_folder=target_folder
            ))
            
            if result.get('status') != 'success':
                return TaskResponse(
                    task_id=task.task_id,
                    status="failed",
                    result={},
                    error=result.get('error', 'Unknown error'),
                    processing_time_seconds=time.time() - start_time,
                    metadata={"agent": "config_collector"}
                )
            
            # Extract file path and metadata (NEW: context_bundle format)
            context_bundle_file = result.get('output_file')
            result_data = result.get('result', {})
            summary = result_data.get('summary', {})
            
            logger.info(f"✅ Config Collector completed: {context_bundle_file}")
            logger.info(f"   Files with drift: {summary.get('files_with_drift', 0)}")
            logger.info(f"   Total deltas: {summary.get('total_deltas', 0)}")
            
            return TaskResponse(
                task_id=task.task_id,
                status="success",
                result={
                    "context_bundle_file": context_bundle_file,  # NEW: context_bundle instead of drift_analysis
                    "summary": summary
                },
                error=None,
                processing_time_seconds=time.time() - start_time,
                metadata={
                    "agent": "config_collector",
                    "context_bundle_file": context_bundle_file,
                    "golden_branch": golden_branch,
                    "drift_branch": drift_branch
                }
            )
            
        except Exception as e:
            logger.exception(f"❌ Config Collector task processing failed: {e}")
            return TaskResponse(
                task_id=task.task_id,
                status="failed",
                result={},
                error=str(e),
                processing_time_seconds=time.time() - start_time,
                metadata={"agent": "config_collector"}
            )
        
    def _get_config_collector_prompt(self) -> str:
        return (
            """You are the Config Collector Agent in the Golden Config AI system.\n"""
            "Your responsibilities:\n"
            "1. Identify configuration files across branches.\n"
            "2. Collect structured diffs between golden and drift branches.\n"
            "3. Preserve exact output schemas used by existing automation.\n"
            "4. Never modify repository state beyond safe read-only diff operations.\n"
            "Return JSON-serializable Python dicts. Maintain existing key names."
        )

    # Tool methods using only serializable types (Dict, List, str)
    @tool
    async def collect_repository_diffs(self, repo_url: str, repo_path: str, 
                                     golden_branch: str, drift_branch: str,
                                     target_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect configuration diffs between golden and drift branches
        
        Args:
            repo_url: Repository URL
            repo_path: Local path to clone/access repository
            golden_branch: Golden/baseline branch name
            drift_branch: Drift branch name to compare
            target_folder: Optional subfolder to analyze
            
        Returns:
            Detailed diff analysis results
        """
        logger.info(f"Collecting diffs between {golden_branch} and {drift_branch}")
        
        try:
            # Use helper functions to do the actual git work
            repo = ensure_repo_ready(repo_url, Path(repo_path))
            if not repo:
                return {
                    "status": "error",
                    "error": "Failed to prepare repository",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            diff_results = collect_branch_diffs(repo, golden_branch, drift_branch, target_folder)
            
            return {
                "status": "success",
                "diff_analysis": diff_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Repository diff collection failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def analyze_config_files(self, repo_url: str, repo_path: str,
                                 target_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze and identify configuration files in repository
        
        Args:
            repo_url: Repository URL
            repo_path: Local path to clone/access repository
            target_folder: Optional subfolder to analyze
            
        Returns:
            List of configuration file paths and analysis
        """
        logger.info(f"Analyzing configuration files in repository")
        
        try:
            repo = ensure_repo_ready(repo_url, Path(repo_path))
            if not repo:
                return {
                    "status": "error",
                    "error": "Failed to prepare repository",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            config_paths = get_config_file_paths(repo, target_folder)
            
            return {
                "status": "success",
                "config_files": config_paths,
                "total_files": len(config_paths),
                "target_folder": target_folder or "entire_repository",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Config file analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def setup_repository_access(self, repo_url: str) -> Dict[str, Any]:
        """
        Setup repository access with authentication
        
        Args:
            repo_url: Repository URL to setup access for
            
        Returns:
            Setup status and authenticated URL
        """
        logger.info(f"Setting up repository access")
        
        try:
            authenticated_url = setup_git_auth(repo_url)
            configure_git_user()
            
            return {
                "status": "success",
                "authenticated_url": authenticated_url,
                "has_auth": authenticated_url != repo_url,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Repository access setup failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    @tool
    async def run_complete_diff_workflow(self, repo_url: str, golden_branch: str = "golden_branch", 
                                       drift_branch: str = "drift_branch", 
                                       target_folder: str = "") -> Dict[str, Any]:
        """
        Run complete drift analysis workflow using drift.py for precision.
    
        This replaces the manual Git diff parsing with drift.py's advanced analysis,
        providing precise line-level locators, structured deltas, and policy-aware tagging.
    
        Args:
            repo_url: Repository URL to analyze
            golden_branch: Golden/reference branch 
            drift_branch: Drift/comparison branch
            target_folder: Specific folder to analyze (empty = entire repo)
        
        Returns:
            Complete drift analysis with context_bundle.json file path
        """
        logger.info("=" * 60)
        logger.info(f"🚀 Starting Drift Analysis with drift.py Precision")
        logger.info(f"   Golden: {golden_branch}")
        logger.info(f"   Drift: {drift_branch}")
        logger.info("=" * 60)
    
        # Create temporary directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_temp = Path(tempfile.gettempdir()) / "golden_config_drift"
        base_temp.mkdir(parents=True, exist_ok=True)
    
        golden_temp = base_temp / f"golden_{timestamp}"
        drift_temp = base_temp / f"drift_{timestamp}"
    
        try:
            # ================================================================
            # PHASE 1: CLONE REPOS (KEEP EXISTING LOGIC)
            # ================================================================
            logger.info("\n📥 Phase 1: Cloning repositories")
            logger.info("-" * 60)
        
            # Configure Git user
            configure_git_user()
        
            # Clone golden branch
            logger.info(f"Cloning golden branch '{golden_branch}'...")
            golden_repo = ensure_repo_ready(repo_url, golden_temp)
            if not golden_repo:
                raise Exception("Failed to setup golden repository")
        
            switch_to_branch(golden_repo, golden_branch)
            logger.info(f"✅ Golden branch ready at: {golden_temp}")
        
            # Clone drift branch
            logger.info(f"Cloning drift branch '{drift_branch}'...")
            drift_repo = ensure_repo_ready(repo_url, drift_temp)
            if not drift_repo:
                raise Exception("Failed to setup drift repository")
        
            switch_to_branch(drift_repo, drift_branch)
            logger.info(f"✅ Drift branch ready at: {drift_temp}")
        
            # ================================================================
            # PHASE 2: DRIFT.PY PRECISION ANALYSIS
            # ================================================================
            logger.info("\n🔍 Phase 2: Running drift.py Precision Analysis")
            logger.info("-" * 60)
        
            # Step 1: Extract file trees
            logger.info("Extracting repository file trees...")
            golden_paths = extract_repo_tree(golden_temp)
            drift_paths = extract_repo_tree(drift_temp)
            logger.info(f"  Golden: {len(golden_paths)} files")
            logger.info(f"  Drift: {len(drift_paths)} files")
            
            # DEBUG: Always show what files we found (for troubleshooting)
            logger.info("\n  📂 Golden files found:")
            for idx, f in enumerate(golden_paths, 1):
                logger.info(f"    {idx}. {f}")
            
            logger.info("\n  📂 Drift files found:")
            for idx, f in enumerate(drift_paths, 1):
                logger.info(f"    {idx}. {f}")
        
            # Step 2: Classify files
            logger.info("Classifying files by type...")
            golden_files = classify_files(golden_temp, golden_paths)
            drift_files = classify_files(drift_temp, drift_paths)
        
            # Step 3: Structural diff
            logger.info("Computing structural diff...")
            file_changes = diff_structural(golden_files, drift_files)
            logger.info(f"  Added: {len(file_changes['added'])} files")
            logger.info(f"  Removed: {len(file_changes['removed'])} files")
            logger.info(f"  Modified: {len(file_changes['modified'])} files")
            logger.info(f"  Renamed: {len(file_changes['renamed'])} files")
        
            # Step 4: Configuration diff (semantic, key-level)
            logger.info("Analyzing configuration changes...")
            changed_paths = sorted(
                set(file_changes["modified"]) | set(file_changes["added"])
            )
            config_diff = semantic_config_diff(
                golden_temp, 
                drift_temp, 
                changed_paths
            )
            logger.info(f"  Config keys added: {len(config_diff.get('added', {}))}")
            logger.info(f"  Config keys removed: {len(config_diff.get('removed', {}))}")
            logger.info(f"  Config keys changed: {len(config_diff.get('changed', {}))}")
        
            # Step 5: Dependency analysis
            logger.info("Analyzing dependency changes...")
            golden_deps = extract_dependencies(golden_temp)
            drift_deps = extract_dependencies(drift_temp)
            dep_diff = dependency_diff(golden_deps, drift_deps)
        
            dep_changes = 0
            for eco, changes in dep_diff.items():
                dep_changes += len(changes.get('added', {}))
                dep_changes += len(changes.get('removed', {}))
                dep_changes += len(changes.get('changed', {}))
            logger.info(f"  Dependency changes: {dep_changes}")
        
            # Step 6: Specialized detectors
            logger.info("Running specialized detectors...")
            spring_deltas = detector_spring_profiles(golden_temp, drift_temp)
            jenkins_deltas = detector_jenkinsfile(golden_temp, drift_temp)
            logger.info(f"  Spring profile deltas: {len(spring_deltas)}")
            logger.info(f"  Jenkinsfile deltas: {len(jenkins_deltas)}")
        
            # Step 7: Code hunks (line-precise diffs)
            logger.info("Building code hunks with line numbers...")
            code_hunks = build_code_hunk_deltas(
                golden_temp,
                drift_temp,
                file_changes.get("modified", [])
            )
            logger.info(f"  Code hunks: {len(code_hunks)}")
        
            # Step 8: Binary file analysis
            logger.info("Analyzing binary files...")
            binary_deltas = build_binary_deltas(
                golden_temp,
                drift_temp,
                file_changes.get("modified", [])
            )
            logger.info(f"  Binary file changes: {len(binary_deltas)}")
        
            # ================================================================
            # PHASE 3: BUILD CONTEXT BUNDLE
            # ================================================================
            logger.info("\n📦 Phase 3: Building Context Bundle")
            logger.info("-" * 60)
        
            # Prepare output directory
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
            output_base = PROJECT_ROOT / "config_data" / "context_bundles"
            output_base.mkdir(parents=True, exist_ok=True)
            output_dir = output_base / f"bundle_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
        
            # Build overview
            overview = {
                "golden_files": len(golden_files),
                "drift_files": len(drift_files),
                "languages_hint": sorted({
                    f["ext"] for f in drift_files 
                    if f["file_type"] == "code"
                }),
                "ci_present": any(
                    "jenkinsfile" in f["name"].lower() 
                    for f in drift_files
                ),
                "build_tools": [
                    f["name"] for f in drift_files 
                    if f["file_type"] == "build"
                ][:10],
                "repo_url": repo_url,
                "golden_branch": golden_branch,
                "drift_branch": drift_branch,
                "timestamp": timestamp,
                "environment": "production"  # TODO: Make this configurable
            }
        
            # Combine all deltas
            extra_deltas = (
                spring_deltas + 
                jenkins_deltas + 
                code_hunks + 
                binary_deltas
            )
        
            # Load policies (optional)
            policies_path = Path(__file__).parent.parent.parent.parent / "shared" / "policies.yaml"
            if not policies_path.exists():
                logger.warning("⚠️  policies.yaml not found, proceeding without policy tagging")
                policies_path = None
            else:
                logger.info(f"✅ Using policies from: {policies_path}")
        
            # Emit context bundle
            logger.info("Generating context_bundle.json...")
            bundle_data = emit_context_bundle(
                output_dir,
                golden_temp,
                drift_temp,
                overview,
                dep_diff,
                config_diff,
                file_changes,
                extra_deltas=extra_deltas,
                policies_path=policies_path,
                evidence=None  # Can add later
            )
        
            context_bundle_path = output_dir / "context_bundle.json"
        
            # ================================================================
            # PHASE 4: PREPARE RESPONSE
            # ================================================================
            logger.info("\n✅ Phase 4: Analysis Complete!")
            logger.info("-" * 60)
        
            # Count files with drift (from deltas)
            with open(context_bundle_path, 'r', encoding='utf-8') as f:
                bundle = json.load(f)
        
            deltas = bundle.get("deltas", [])
            files_with_drift = len(set(d.get("file", "") for d in deltas))
        
            # Summary
            summary = {
                "files_compared": len(golden_paths),
                "files_with_drift": files_with_drift,
                "total_deltas": len(deltas),
                "config_changes": len(config_diff.get("changed", {})),
                "dependency_changes": dep_changes,
                "code_hunks": len(code_hunks),
                "policies_applied": policies_path is not None
            }
        
            logger.info("📊 Summary:")
            logger.info(f"   Files compared: {summary['files_compared']}")
            logger.info(f"   Files with drift: {summary['files_with_drift']}")
            logger.info(f"   Total deltas: {summary['total_deltas']}")
            logger.info(f"   Config changes: {summary['config_changes']}")
            logger.info(f"   Dependency changes: {summary['dependency_changes']}")
            logger.info(f"   Code hunks: {summary['code_hunks']}")
            logger.info(f"   Output: {context_bundle_path}")
            logger.info("=" * 60)
        
            return {
                "status": "success",
                "output_file": str(context_bundle_path),
                "result": {
                    "context_bundle_file": str(context_bundle_path),
                    "summary": summary,
                    "overview": overview
                },
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.exception(f"❌ Error in drift analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # Cleanup temp directories
            logger.info("🧹 Cleaning up temporary directories...")
            if golden_temp.exists():
                shutil.rmtree(golden_temp, ignore_errors=True)
            if drift_temp.exists():
                shutil.rmtree(drift_temp, ignore_errors=True)
            logger.info("✅ Cleanup complete")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Collect configuration diffs between branches in gc-cxp repository")
    parser.add_argument("--golden-branch", default="golden_branch", help="Golden/reference branch (default: golden_branch)")
    parser.add_argument("--drift-branch", default="drift_branch", help="Drift/comparison branch (default: drift_branch)")
    parser.add_argument("--target-folder", default="cxp-ordering-services", help="Specific folder to analyze (default: cxp-ordering-services)")
    parser.add_argument("--validate-paths", action="store_true", help="Validate path resolution and exit (no git, no scanning)")
    
    args = parser.parse_args()
    
    # --- CONFIGURATION ---
    TARGET_REPO_URL = 'https://gitlab.verizon.com/yadvi7z/gc-cxp.git'
    
    # --- PATH SETUP (UPDATED FOR TEMP DIRS AND STRANDS STORAGE) ---
    # Current location is golden-config-ai-poc/strands-multi-agent-system/agents/workers/config_collector/
    try:
        # Get strands project root (strands-multi-agent-system/)
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()  # strands-multi-agent-system/
        print(f"[INFO] Project root: {PROJECT_ROOT}")
    except NameError:
        # Fallback for interactive environments
        PROJECT_ROOT = Path('.').resolve()

    # Create temporary directory for repository
    temp_dir = tempfile.mkdtemp(prefix="config_collector_")
    TARGET_REPO_LOCAL_PATH = Path(temp_dir) / "repo"
    DATA_DIR = PROJECT_ROOT / "config_data"
    DRIFT_ANALYSIS_DIR = DATA_DIR / "drift_analysis"
    
    if args.validate_paths:
        print("\n--- Path Validation (config_collector_agent.py) ---")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Target repo path (temporary): {TARGET_REPO_LOCAL_PATH}")
        print(f"Data dir: {DATA_DIR}")
        print(f"Drift analysis dir: {DRIFT_ANALYSIS_DIR}")
        print(f"Golden branch (no checkout performed): {args.golden_branch}")
        print(f"Drift branch (no checkout performed): {args.drift_branch}")
        print(f"Target folder (no scan performed): {args.target_folder}")
        print("✅ Path logic OK (no git operations executed)")
        # Clean up temp dir
        shutil.rmtree(temp_dir)
        sys.exit(0)
    
    try:
        # --- SCRIPT START ---
        print("\n--- Starting Configuration Diff Collection (Agent Version) ---")
        print(f"Golden branch: {args.golden_branch}")
        print(f"Drift branch: {args.drift_branch}")
        print(f"Target folder: {args.target_folder}")
        print(f"Target repo URL: {TARGET_REPO_URL}")
        print(f"Using temporary directory: {temp_dir}")
        
        # 1. Setup Git User and Authentication
        configure_git_user()
        
        # 2. Ensure Target Repository is Cloned and Up-to-Date (in temp location)
        repo = ensure_repo_ready(TARGET_REPO_URL, TARGET_REPO_LOCAL_PATH)
        if not repo:
            print("❌ FATAL: Could not access repository. Exiting.")
            sys.exit(1)
        
        # 3. Collect Branch Diffs
        diff_results = collect_branch_diffs(repo, args.golden_branch, args.drift_branch, args.target_folder)
        
        if "error" in diff_results:
            print(f"❌ Error occurred during diff collection: {diff_results['error']}")
            sys.exit(1)
        
        # 4. Save Results
        DRIFT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"branch_diff_{args.golden_branch}_vs_{args.drift_branch}_{timestamp}.json"
        output_file_path = DRIFT_ANALYSIS_DIR / output_filename
        
        # Save to JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(diff_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- Diff Collection Complete ---")
        print(f"✅ Analyzed {diff_results['summary']['total_files_analyzed']} configuration files")
        print(f"📁 Results saved to: {output_file_path}")
        print(f"\n📊 Summary:")
        print(f"   • Golden branch: {args.golden_branch}")
        print(f"   • Drift branch: {args.drift_branch}")
        print(f"   • Target folder: {args.target_folder or 'entire repository'}")
        print(f"   • Files analyzed: {diff_results['summary']['total_files_analyzed']}")
        print(f"   • Files modified: {len(diff_results['branch_comparison']['files_modified'])}")
        print(f"   • Files added in drift: {diff_results['summary']['files_added']}")
        print(f"   • Files removed in drift: {diff_results['summary']['files_removed']}")
        print(f"   • Lines added: {diff_results['summary']['lines_added']}")
        print(f"   • Lines removed: {diff_results['summary']['lines_removed']}")
        
        # Show changed files as examples
        if diff_results["branch_comparison"]["files_changed"]:
            print(f"\n📝 Changed files:")
            for i, file_info in enumerate(diff_results["branch_comparison"]["files_changed"][:10], 1):
                print(f"   {i}. [{file_info['change_type'].upper()}] {file_info['file_path']}")
            if len(diff_results["branch_comparison"]["files_changed"]) > 10:
                print(f"   ... and {len(diff_results['branch_comparison']['files_changed']) - 10} more files")
        
        return True
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[INFO] Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"[WARN] Could not clean up temporary directory {temp_dir}: {e}")


if __name__ == "__main__":
    main()

