# Config Collector Agent

The **Config Collector Agent** is responsible for fetching configuration diffs from Git repositories and identifying configuration changes between branches.

---

## 🎯 Responsibilities

1. **Git Operations** - Clone repositories and switch branches
2. **Diff Collection** - Extract changes between golden and drift branches
3. **Config Detection** - Identify configuration files (YAML, JSON, XML, Properties)
4. **Change Analysis** - Calculate additions, deletions, and modifications
5. **File Output** - Save drift analysis to JSON files

---

## 🤖 Configuration

**Model:** Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`)

**Why Haiku?** Config Collector performs deterministic tasks:
- Git operations
- File parsing
- Diff extraction
- Does not require complex reasoning

---

## 🔧 Tools

The Config Collector Agent has several specialized tools:

### 1. `clone_repository`
Clones a Git repository to a temporary directory.

**Usage:**
```python
clone_repository(
    repository_url="https://gitlab.verizon.com/saja9l7/golden_config.git",
    branch_name="gold"
)
```

### 2. `switch_branch`
Switches to a different branch in cloned repository.

### 3. `get_branch_diff`
Gets the diff between two branches.

**Returns:**
- List of changed file paths
- Diff content for each file

### 4. `identify_config_files`
Identifies configuration files from changed files.

**Detects:**
- YAML (`.yml`, `.yaml`)
- JSON (`.json`)
- Properties (`.properties`, `.ini`, `.cfg`, `.conf`)
- XML (`.xml`)

### 5. `run_complete_diff_workflow`
Executes the complete drift analysis workflow.

**Workflow:**
1. Clone golden branch
2. Clone drift branch
3. Get diff between branches
4. Identify config files
5. Analyze changes
6. Save to `drift_analysis_*.json`

---

## 📊 Workflow

### Complete Drift Analysis Flow:

```
1. User Request (via Supervisor)
   ↓
2. Clone golden branch
   ├─ git clone <repo_url>
   └─ git checkout <golden_branch>
   ↓
3. Clone drift branch
   ├─ git clone <repo_url>
   └─ git checkout <drift_branch>
   ↓
4. Get diff between branches
   └─ git diff golden..drift
   ↓
5. Identify configuration files
   └─ Filter: .yml, .json, .properties, .xml
   ↓
6. Analyze each file
   ├─ Calculate additions/deletions
   ├─ Determine change type
   └─ Extract diff content
   ↓
7. Save results
   └─ config_data/drift_analysis/drift_analysis_*.json
   ↓
8. Return file path to Supervisor
```

---

## 🚀 Usage

### Via Supervisor (Recommended):

The Supervisor automatically calls this agent as part of the workflow.

### Direct Usage:

```python
from Agents.workers.config_collector.config_collector_agent import ConfigCollectorAgent
from shared.config import Config
from shared.models import TaskRequest

# Initialize
config = Config()
agent = ConfigCollectorAgent(config)

# Create task
task = TaskRequest(
    task_id="test_collection",
    task_type="collect_diffs",
    parameters={
        "repo_url": "https://gitlab.verizon.com/saja9l7/golden_config.git",
        "golden_branch": "main",
        "drift_branch": "drifted",
        "target_folder": ""  # Optional
    }
)

# Process task
result = agent.process_task(task)

print(f"Status: {result.status}")
print(f"Drift file: {result.result['drift_analysis_file']}")
print(f"Files with drift: {result.result['summary']['files_with_drift']}")
```

### Async Usage:

```python
import asyncio

result = await agent.run_complete_diff_workflow(
    repo_url="https://gitlab.verizon.com/saja9l7/golden_config.git",
    golden_branch="gold",
    drift_branch="drift",
    target_folder=""
)

print(f"Output file: {result['output_file']}")
```

---

## 📁 Outputs

### Drift Analysis File
**Location:** `config_data/drift_analysis/drift_analysis_YYYYMMDD_HHMMSS.json`

**Structure:**
```json
{
  "timestamp": "2025-10-04T16:51:33.123456",
  "status": "success",
  "result": {
    "drift_analysis": {
      "total_files_compared": 10,
      "files_with_drift": 3,
      "files_without_changes": 7,
      "detailed_drifts": [
        {
          "file_path": "config/application.yml",
          "has_changes": true,
          "change_type": "modification",
          "diff_content": [
            "@@ -10,7 +10,7 @@",
            " server:",
            "-  port: 8080",
            "+  port: 8090",
            "   host: localhost"
          ],
          "changes": {
            "additions": 1,
            "deletions": 1
          }
        }
      ]
    }
  },
  "metadata": {
    "agent": "config_collector",
    "repo_url": "https://gitlab.verizon.com/saja9l7/golden_config.git",
    "golden_branch": "main",
    "drift_branch": "drifted"
  }
}
```

---

## 🔍 Change Types

The agent detects 3 types of changes:

### 1. **Addition**
New file in drift branch that doesn't exist in golden.

### 2. **Deletion**
File exists in golden but removed in drift.

### 3. **Modification**
File exists in both branches but content differs.

---

## 📊 Change Statistics

For each file, the agent calculates:

- **Additions:** Number of lines added
- **Deletions:** Number of lines deleted
- **Has Changes:** Boolean flag

**Calculation:**
```python
additions = diff_text.count('\n+') - diff_text.count('\n+++')
deletions = diff_text.count('\n-') - diff_text.count('\n---')
```

---

## 🔧 Configuration

Set in `.env`:

```bash
# Worker Model (Claude 3 Haiku)
BEDROCK_WORKER_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# AWS Region
AWS_REGION=us-east-1

# GitLab (for private repos)
GITLAB_PRIVATE_TOKEN=your_token_here
```

---

## 🎯 Supported File Types

### Configuration Files:
- **YAML:** `.yml`, `.yaml`
- **JSON:** `.json`
- **Properties:** `.properties`, `.ini`, `.cfg`, `.conf`
- **XML:** `.xml`

### File Detection:
Files are identified by extension. Custom extensions can be added in the code.

---

## 🐛 Troubleshooting

### "Repository clone failed"
**Check:**
1. Repository URL is accessible
2. GitLab token is valid (for private repos)
3. Git is installed
4. Network connectivity

### "Branch not found"
**Check:**
1. Branch name is correct
2. Branch exists in repository
3. Repository access permissions

### "No config files found"
**Check:**
1. Target folder path is correct
2. Repository contains configuration files
3. File extensions match supported types

### "Diff extraction failed"
**Check:**
1. Both branches cloned successfully
2. Git diff command available
3. Repository is not corrupted

---

## 📚 Related

- **Supervisor:** `../../Supervisor/README.md`
- **Diff Engine:** `../diff_policy_engine/README.md`
- **Main README:** `../../../README.md`

---

**The data collection engine of your multi-agent system!** 📊

