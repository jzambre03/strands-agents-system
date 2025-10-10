"""
Simple GitLab Access Script

Verifies access to GitLab repositories using personal access tokens.
All configuration loaded from .env file.

Usage:
    python gitlab_access.py
"""

import os
import sys
import requests
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env_config():
    """Load all configuration from .env file."""
    config = {
        'gitlab_token': os.getenv('GITLAB_TOKEN'),
        'gitlab_url': os.getenv('GITLAB_URL'),
        'gitlab_base_url': os.getenv('GITLAB_BASE_URL', 'https://gitlab.com'),
        'gitlab_username': os.getenv('GITLAB_USERNAME'),
        'gitlab_password': os.getenv('GITLAB_PASSWORD'),
    }
    return config


def parse_repo_url(repo_url: str, base_url: str = "https://gitlab.com"):
    """
    Parse GitLab repository URL to extract project path.
    
    Args:
        repo_url: Full repository URL (e.g., https://gitlab.com/group/project.git)
        base_url: GitLab base URL (e.g., https://gitlab.com)
    
    Example: https://gitlab.com/group/project.git -> group/project
    """
    # Remove .git suffix if present
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    # Extract hostname from base_url
    # https://gitlab.com -> gitlab.com
    # https://gitlab.example.com:8080 -> gitlab.example.com:8080
    from urllib.parse import urlparse
    parsed_base = urlparse(base_url)
    gitlab_host = parsed_base.netloc if parsed_base.netloc else parsed_base.path
    
    # Extract project path from URL
    if gitlab_host in repo_url:
        parts = repo_url.split(f'{gitlab_host}/')
        if len(parts) > 1:
            return parts[1]
    
    return None


def test_gitlab_access(token: str, repo_url: str, base_url: str = "https://gitlab.com"):
    """
    Test GitLab access using the API without cloning.
    
    Args:
        token: GitLab personal access token
        repo_url: Repository URL
        base_url: GitLab base URL
        
    Returns:
        True if access successful, False otherwise
    """
    print("\n🔍 Testing GitLab Access...")
    print(f"   Repository: {repo_url}")
    
    if not token:
        print("❌ No GITLAB_TOKEN found in .env file")
        return False
    
    # Parse project path from URL
    project_path = parse_repo_url(repo_url, base_url)
    if not project_path:
        print("❌ Could not parse project path from URL")
        return False
    
    # URL encode the project path
    import urllib.parse
    encoded_path = urllib.parse.quote(project_path, safe='')
    
    # GitLab API endpoint
    api_url = f"{base_url}/api/v4/projects/{encoded_path}"
    
    headers = {
        'PRIVATE-TOKEN': token
    }
    
    try:
        # Test API access
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Access successful!")
            print("\n📊 Repository Information:")
            print(f"   Name: {data.get('name', 'N/A')}")
            print(f"   Description: {data.get('description', 'N/A')}")
            print(f"   Default Branch: {data.get('default_branch', 'N/A')}")
            print(f"   Visibility: {data.get('visibility', 'N/A')}")
            print(f"   Created: {data.get('created_at', 'N/A')}")
            print(f"   Last Activity: {data.get('last_activity_at', 'N/A')}")
            
            # Get branches
            branches_url = f"{api_url}/repository/branches"
            branches_response = requests.get(branches_url, headers=headers)
            
            if branches_response.status_code == 200:
                branches = branches_response.json()
                print(f"\n📋 Branches ({len(branches)}):")
                for branch in branches[:10]:  # Show first 10
                    print(f"   - {branch['name']}")
                if len(branches) > 10:
                    print(f"   ... and {len(branches) - 10} more")
            
            return True
        
        elif response.status_code == 401:
            print("❌ Authentication failed - Invalid token")
            return False
        
        elif response.status_code == 404:
            print("❌ Repository not found or no access")
            return False
        
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing GitLab API: {e}")
        return False


def create_gitlab_project(token: str, project_name: str, namespace: str = None, visibility: str = "private", base_url: str = "https://gitlab.com"):
    """
    Create a new GitLab project/repository.
    
    Args:
        token: GitLab personal access token
        project_name: Name for the new project
        namespace: Group/namespace (e.g., 'group-name'). If None, creates in user's namespace
        visibility: 'private', 'internal', or 'public'
        
    Returns:
        dict with project info if successful, None otherwise
    """
    print(f"\n🔨 Creating new GitLab project: {project_name}")
    
    api_url = f"{base_url}/api/v4/projects"
    headers = {
        'PRIVATE-TOKEN': token,
        'Content-Type': 'application/json'
    }
    
    data = {
        'name': project_name,
        'visibility': visibility,
        'initialize_with_readme': False
    }
    
    # Add namespace if provided
    if namespace:
        # Get namespace ID
        namespace_url = f"{base_url}/api/v4/namespaces?search={namespace}"
        ns_response = requests.get(namespace_url, headers=headers)
        
        if ns_response.status_code == 200 and ns_response.json():
            namespace_id = ns_response.json()[0]['id']
            data['namespace_id'] = namespace_id
            print(f"   Using namespace: {namespace} (ID: {namespace_id})")
        else:
            print(f"⚠️  Warning: Namespace '{namespace}' not found, using user's namespace")
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        
        if response.status_code == 201:
            project_data = response.json()
            print(f"✅ Project created successfully!")
            print(f"   Project ID: {project_data['id']}")
            print(f"   URL: {project_data['web_url']}")
            print(f"   Git URL: {project_data['http_url_to_repo']}")
            return project_data
        
        elif response.status_code == 400:
            error_msg = response.json().get('message', {})
            if 'has already been taken' in str(error_msg):
                print(f"⚠️  Project '{project_name}' already exists")
                # Try to get existing project
                search_url = f"{base_url}/api/v4/projects?search={project_name}"
                search_response = requests.get(search_url, headers=headers)
                if search_response.status_code == 200:
                    projects = search_response.json()
                    for proj in projects:
                        if proj['name'] == project_name:
                            print(f"   Using existing project: {proj['http_url_to_repo']}")
                            return proj
            print(f"❌ Error: {error_msg}")
            return None
        
        else:
            print(f"❌ Failed to create project: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating project: {e}")
        return None


def copy_config_files_to_new_repo(
    source_repo_url: str,
    new_project_name: str,
    token: str,
    namespace: str = None,
    config_extensions: List[str] = None,
    keep_history: bool = True,
    base_url: str = "https://gitlab.com",
    visibility: str = "private",
    target_branch: str = "main",
    max_file_size_mb: int = 10,
    max_total_size_mb: int = 100,
    create_placeholders_for_skipped: bool = True
):
    """
    Create a new GitLab repo and copy only config files from source repo using API (no cloning).
    
    Uses GitLab Search API to find files by extension - extremely fast even for large repos!
    Instead of fetching all files (could be 10k+), only searches for specified config extensions.
    
    Args:
        source_repo_url: URL of source repository
        new_project_name: Name for new project
        token: GitLab personal access token
        namespace: Optional namespace/group for new project
        config_extensions: List of config file extensions to copy (e.g., ['yml', 'properties'])
        keep_history: If True, preserves Git history (not implemented in API version)
        base_url: GitLab instance URL
        visibility: Project visibility ('private', 'internal', 'public')
        target_branch: Branch to commit to
        max_file_size_mb: Skip files larger than this (MB)
        max_total_size_mb: Warn if total memory usage exceeds this (MB)
        create_placeholders_for_skipped: Create .SKIPPED.md files for large files
        
    Returns:
        dict with result info
    """
    if config_extensions is None:
        config_extensions = ['yml', 'yaml', 'properties', 'conf', 'config', 'toml', 'env']
    
    print("=" * 70)
    print("🚀 Config Files Migration to New Repository (Optimized)")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Source: {source_repo_url}")
    print(f"   New Project: {new_project_name}")
    print(f"   Config Extensions: {', '.join(config_extensions)}")
    print(f"   Method: GitLab Search API (fast, targeted, no disk download)")
    print(f"   Performance: Only fetches {len(config_extensions)} extensions (not entire repo tree)")
    
    try:
        # Parse source project path
        source_project_path = parse_repo_url(source_repo_url, base_url)
        if not source_project_path:
            return {"status": "error", "message": "Could not parse source repository URL"}
        
        import urllib.parse
        source_project_id = urllib.parse.quote(source_project_path, safe='')
        
        headers = {'PRIVATE-TOKEN': token}
        
        # Step 1: Create new GitLab project
        print("\n" + "=" * 70)
        print("Step 1: Creating New GitLab Project")
        print("=" * 70)
        
        new_project = create_gitlab_project(token, new_project_name, namespace, visibility, base_url)
        if not new_project:
            return {"status": "error", "message": "Failed to create new project"}
        
        new_project_id = new_project['id']
        new_repo_url = new_project['http_url_to_repo']
        
        # Step 2: Search for config files by extension (optimized - no full tree fetch!)
        print("\n" + "=" * 70)
        print("Step 2: Searching for Config Files (by extension)")
        print("=" * 70)
        
        print(f"   Searching for extensions: {', '.join(config_extensions)}")
        print(f"   Using GitLab search API (much faster than fetching all files)...")
        
        config_files = []
        seen_paths = set()  # Avoid duplicates
        
        # Search for files matching each extension
        for ext in config_extensions:
            print(f"   Searching for *.{ext} files...")
            search_url = f"{base_url}/api/v4/projects/{source_project_id}/search"
            
            page = 1
            per_page = 100
            ext_count = 0
            
            while True:
                response = requests.get(
                    search_url,
                    headers=headers,
                    params={
                        'scope': 'blobs',
                        'search': f'filename:*.{ext}',
                        'per_page': per_page,
                        'page': page
                    }
                )
                
                if response.status_code != 200:
                    print(f"   ⚠️  Warning: Search failed for *.{ext}: {response.status_code}")
                    break
                
                results = response.json()
                if not results:
                    break
                
                for result in results:
                    file_path = result.get('path') or result.get('filename')
                    
                    # Skip if we've already seen this file (avoid duplicates)
                    if file_path in seen_paths:
                        continue
                    
                    # Skip .git directory
                    if file_path and file_path.startswith('.git/'):
                        continue
                    
                    # Verify file actually ends with the extension
                    if file_path and file_path.endswith(f'.{ext}'):
                        config_files.append({
                            'path': file_path,
                            'type': 'blob'
                        })
                        seen_paths.add(file_path)
                        ext_count += 1
                
                # Check if there are more pages
                if len(results) < per_page:
                    break
                page += 1
            
            if ext_count > 0:
                print(f"      ✅ Found {ext_count} *.{ext} files")
        
        print(f"\n✅ Found {len(config_files)} config files total (matching extensions: {', '.join(config_extensions)}):")
        for i, file in enumerate(config_files[:10], 1):
            print(f"   {i}. {file['path']}")
        if len(config_files) > 10:
            print(f"   ... and {len(config_files) - 10} more")
        
        if not config_files:
            print("⚠️  No config files found!")
            return {"status": "error", "message": "No config files found"}
        
        # Step 4: Get file contents
        print("\n" + "=" * 70)
        print("Step 4: Fetching Config File Contents (to memory)")
        print("=" * 70)
        
        file_actions = []
        folders = set()
        skipped_large_files = []
        total_size_mb = 0
        
        # Safety limits (configurable)
        MAX_FILE_SIZE_MB = max_file_size_mb
        MAX_TOTAL_SIZE_MB = max_total_size_mb
        
        for i, file in enumerate(config_files, 1):
            file_path = file['path']
            encoded_path = urllib.parse.quote(file_path, safe='')
            
            # Get file content
            content_url = f"{base_url}/api/v4/projects/{source_project_id}/repository/files/{encoded_path}/raw"
            
            try:
                content_response = requests.get(content_url, headers=headers)
                
                if content_response.status_code == 200:
                    content = content_response.text
                    
                    # Check file size
                    file_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
                    
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        print(f"   ⚠️  Skipping {file_path} (size: {file_size_mb:.2f}MB exceeds {MAX_FILE_SIZE_MB}MB limit)")
                        skipped_large_files.append({
                            'path': file_path, 
                            'size_mb': file_size_mb,
                            'url': content_url
                        })
                        
                        # Create placeholder file explaining why it was skipped
                        if create_placeholders_for_skipped:
                            placeholder_content = f"""# FILE TOO LARGE - NOT MIGRATED

This file was skipped during automated migration because it exceeds the size limit.

**Original File:** {file_path}
**File Size:** {file_size_mb:.2f} MB
**Size Limit:** {MAX_FILE_SIZE_MB} MB
**Source Repository:** {source_repo_url}

## How to Migrate This File Manually

### Option 1: Increase the limit and re-run
Update your .env file:
```bash
MAX_FILE_SIZE_MB=50  # Or higher
```

Then run migration again with a different project name.

### Option 2: Download and upload manually
```bash
# Download from source
curl -H "PRIVATE-TOKEN: your_token" "{content_url}" > {file_path}

# Add to this repository
git add {file_path}
git commit -m "Add large config file: {file_path}"
git push
```

### Option 3: Use Git LFS (for very large files)
```bash
git lfs install
git lfs track "{file_path}"
git add .gitattributes {file_path}
git commit -m "Add large file with LFS"
git push
```

**Note:** Config files this large may indicate they contain data that 
should be stored elsewhere (database, object storage, etc.)
"""
                            file_actions.append({
                                'action': 'create',
                                'file_path': f"{file_path}.SKIPPED.md",
                                'content': placeholder_content
                            })
                            print(f"      → Created placeholder: {file_path}.SKIPPED.md")
                        
                        continue
                    
                    total_size_mb += file_size_mb
                    
                    # Warn if approaching memory limit
                    if total_size_mb > MAX_TOTAL_SIZE_MB:
                        print(f"\n⚠️  WARNING: Total size ({total_size_mb:.2f}MB) exceeds recommended limit ({MAX_TOTAL_SIZE_MB}MB)")
                        print(f"   This may cause memory issues. Consider running in batches.")
                    
                    file_actions.append({
                        'action': 'create',
                        'file_path': file_path,
                        'content': content
                    })
                    
                    # Track folders
                    folder = str(Path(file_path).parent)
                    if folder != '.':
                        folders.add(folder)
                    
                    if i % 10 == 0 or i == len(config_files):
                        print(f"   Fetched {i}/{len(config_files)} files (total: {total_size_mb:.2f}MB in memory)...")
                else:
                    print(f"   ⚠️  Warning: Could not fetch {file_path} (status: {content_response.status_code})")
                    
            except Exception as e:
                print(f"   ⚠️  Warning: Error fetching {file_path}: {e}")
        
        print(f"✅ Successfully fetched {len(file_actions)} config files (total: {total_size_mb:.2f}MB in memory)")
        
        if skipped_large_files:
            print(f"\n⚠️  Skipped {len(skipped_large_files)} large files (see .SKIPPED.md files for instructions):")
            for skipped in skipped_large_files[:5]:
                print(f"   - {skipped['path']} ({skipped['size_mb']:.2f}MB)")
            if len(skipped_large_files) > 5:
                print(f"   ... and {len(skipped_large_files) - 5} more")
            print(f"\n💡 To migrate skipped files:")
            print(f"   1. Increase MAX_FILE_SIZE_MB in .env")
            print(f"   2. Or check the .SKIPPED.md files for manual migration steps")
        
        # Step 5: Batch commit all files to new repository
        print("\n" + "=" * 70)
        print("Step 5: Committing Files to New Repository")
        print("=" * 70)
        
        # GitLab has a limit of ~100 actions per commit, so batch if needed
        batch_size = 100
        total_batches = (len(file_actions) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(file_actions))
            batch_actions = file_actions[start_idx:end_idx]
            
            commit_data = {
                'branch': target_branch,
                'commit_message': f'Add config files (batch {batch_num + 1}/{total_batches})' if total_batches > 1 else f'Initial commit: Config files from {source_repo_url}',
                'actions': batch_actions
            }
            
            commit_url = f"{base_url}/api/v4/projects/{new_project_id}/repository/commits"
            
            print(f"   Committing batch {batch_num + 1}/{total_batches} ({len(batch_actions)} files)...")
            
            commit_response = requests.post(
                commit_url,
                headers={**headers, 'Content-Type': 'application/json'},
                json=commit_data
            )
            
            if commit_response.status_code not in [200, 201]:
                error_msg = commit_response.json().get('message', 'Unknown error')
                print(f"❌ Commit failed: {error_msg}")
                return {"status": "error", "message": f"Failed to commit files: {error_msg}"}
        
        print(f"✅ Successfully committed all files to new repository!")
        
        # Step 6: Summary
        print("\n" + "=" * 70)
        print("✅ Migration Complete!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   Config files found: {len(config_files)}")
        print(f"   Config files copied: {len(file_actions)}")
        print(f"   Folders created: {len(folders)}")
        if skipped_large_files:
            print(f"   Large files skipped: {len(skipped_large_files)}")
        print(f"   New repository: {new_repo_url}")
        print(f"   Project URL: {new_project['web_url']}")
        print(f"\n💡 Note: All operations via GitLab Search API - no full tree fetch, no disk writes")
        
        return {
            "status": "success",
            "files_copied": len(file_actions),
            "folders_created": len(folders),
            "config_files_found": len(config_files),
            "large_files_skipped": len(skipped_large_files),
            "new_repo_url": new_repo_url,
            "project_url": new_project['web_url']
        }
        
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def main():
    """Main function."""
    print("=" * 70)
    print("🚀 GitLab Access Verification")
    print("=" * 70)
    
    # Load configuration from .env
    config = get_env_config()
    
    # Check for required variables
    if not config['gitlab_token']:
        print("\n❌ Missing GITLAB_TOKEN in .env file")
        print("   Please add: GITLAB_TOKEN=your_token_here")
        sys.exit(1)
    
    if not config['gitlab_url']:
        print("\n❌ Missing GITLAB_URL in .env file")
        print("   Please add: GITLAB_URL=https://gitlab.com/user/repo.git")
        sys.exit(1)
    
    print("\n✅ Configuration loaded from .env:")
    print(f"   GITLAB_TOKEN: {'*' * 20} (hidden)")
    print(f"   GITLAB_BASE_URL: {config['gitlab_base_url']}")
    print(f"   GITLAB_URL: {config['gitlab_url']}")
    
    # Test access
    success = test_gitlab_access(config['gitlab_token'], config['gitlab_url'], config['gitlab_base_url'])
    
    if success:
        print("\n" + "=" * 70)
        print("✅ GitLab access verified successfully!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ GitLab access verification failed")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
