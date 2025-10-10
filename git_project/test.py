"""
Config file migration to new GitLab repository.

All configuration loaded from .env file.
"""

import os
import sys
from dotenv import load_dotenv
from gitlab_access import copy_config_files_to_new_repo

# Load environment variables
load_dotenv()


def migrate_configs():
    """
    Migrate config files from source repo to new repo.
    All settings loaded from .env file.
    """
    # Get all configuration from .env
    token = os.getenv('GITLAB_TOKEN')
    source_repo_url = os.getenv('GITLAB_SOURCE_URL')
    new_project_name = os.getenv('NEW_PROJECT_NAME')
    namespace = os.getenv('GITLAB_NAMESPACE')
    base_url = os.getenv('GITLAB_BASE_URL', 'https://gitlab.com')
    visibility = os.getenv('PROJECT_VISIBILITY', 'private')
    target_branch = os.getenv('TARGET_BRANCH', 'main')
    config_extensions = os.getenv('CONFIG_EXTENSIONS', 'yml,yaml,properties,conf,config,toml,env')
    keep_history = os.getenv('KEEP_HISTORY', 'true').lower() == 'true'
    
    # Validate required variables
    if not token:
        print("❌ Missing GITLAB_TOKEN in .env file")
        sys.exit(1)
    
    if not source_repo_url:
        print("❌ Missing GITLAB_SOURCE_URL in .env file")
        sys.exit(1)
    
    if not new_project_name:
        print("❌ Missing NEW_PROJECT_NAME in .env file")
        sys.exit(1)
    
    # Parse config extensions
    extensions_list = [ext.strip() for ext in config_extensions.split(',')]
    
    print("=" * 70)
    print("🔧 Configuration loaded from .env:")
    print("=" * 70)
    print(f"   GITLAB_TOKEN: {'*' * 20} (hidden)")
    print(f"   GITLAB_BASE_URL: {base_url}")
    print(f"   GITLAB_SOURCE_URL: {source_repo_url}")
    print(f"   NEW_PROJECT_NAME: {new_project_name}")
    print(f"   GITLAB_NAMESPACE: {namespace or '(user namespace)'}")
    print(f"   PROJECT_VISIBILITY: {visibility}")
    print(f"   TARGET_BRANCH: {target_branch}")
    print(f"   CONFIG_EXTENSIONS: {', '.join(extensions_list)}")
    print(f"   KEEP_HISTORY: {keep_history}")
    print("=" * 70)
    
    # Run migration
    result = copy_config_files_to_new_repo(
        source_repo_url=source_repo_url,
        new_project_name=new_project_name,
        token=token,
        namespace=namespace,
        config_extensions=extensions_list,
        keep_history=keep_history,
        base_url=base_url,
        visibility=visibility,
        target_branch=target_branch
    )
    
    # Check result
    if result['status'] == 'success':
        print("\n✅ Migration successful!")
        print(f"   New repository: {result['new_repo_url']}")
        print(f"   Files copied: {result['files_copied']}")
        return 0
    else:
        print(f"\n❌ Migration failed: {result.get('message', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(migrate_configs())

