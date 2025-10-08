# Diff Policy Engine Agent

The **Diff Policy Engine Agent** performs AI-powered analysis of configuration drift, identifies policy violations, assesses risks, and provides actionable recommendations.

---

## 🎯 Responsibilities

1. **AI-Powered Analysis** - Intelligent drift analysis using enhanced prompts
2. **Risk Assessment** - Per-file and overall risk evaluation
3. **Policy Validation** - Detect security, compliance, and operational violations
4. **Recommendations** - Generate actionable remediation steps
5. **File Output** - Save comprehensive analysis to JSON files

---

## 🤖 Configuration

**Model:** Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`)

**Why Haiku?** 
- Fast analysis for per-file drift assessment
- Cost-effective for multiple file analysis
- Sufficient reasoning for policy validation
- Enhanced with context-rich prompts

---

## 🔧 Tools

The Diff Policy Engine Agent has 6 specialized tools:

### 1. `analyze_configuration_drift`
**The core analysis tool with enhanced AI prompts.**

**Context Provided to AI:**
- Environment (production, staging, development)
- File path and configuration type
- Change statistics (additions/deletions)
- Change type (addition, deletion, modification)
- File importance explanation
- Example output format

**Returns:**
```json
{
  "risk_level": "high",
  "change_impact": "Disables SSL encryption...",
  "policy_violations": [
    {
      "type": "security",
      "severity": "critical",
      "description": "SSL/TLS must be enforced",
      "rule": "SEC_TLS_REQUIRED"
    }
  ],
  "recommendations": [
    {
      "priority": "immediate",
      "action": "Revert SSL disablement",
      "rationale": "Unencrypted traffic exposes data"
    }
  ],
  "security_impact": "CRITICAL - All traffic exposed",
  "operational_impact": "HIGH - Rollback required"
}
```

### 2. `assess_risk_level`
Basic risk calculation based on change patterns.

**Risk Levels:**
- **Critical:** Security/auth changes, sensitive configs
- **High:** Major config modifications, production changes
- **Medium:** Standard config updates
- **Low:** Minor changes, documentation

### 3. `check_policy_violations`
Validates changes against security and compliance policies.

**Checks:**
- Security settings (SSL, encryption, authentication)
- Compliance requirements
- Best practices adherence
- Operational stability risks

### 4. `assess_overall_drift_risk`
**AI-powered holistic risk assessment across all files.**

**Considers:**
- Environment criticality (production = higher risk)
- Change volume and complexity
- Security implications
- Business impact
- Risk distribution

**Returns:**
```json
{
  "overall_risk_level": "high",
  "risk_factors": [
    "5 high-risk changes in production",
    "3 security violations",
    "Large change volume increases deployment risk"
  ],
  "mitigation_strategies": [
    "Conduct immediate review of high-risk changes",
    "Resolve security violations before deployment",
    "Perform comprehensive testing in staging"
  ],
  "mitigation_priority": "urgent"
}
```

### 5. `generate_recommendations`
Creates actionable recommendations based on analysis.

**Recommendations Include:**
- Priority (immediate, high, medium, low)
- Specific action to take
- Rationale for the recommendation

---

## 📊 Workflow

### Complete Analysis Flow:

```
1. Receive drift_analysis file path from Config Collector
   ↓
2. Load drift_analysis_*.json
   └─ Parse detailed_drifts array
   ↓
3. For each file with changes:
   │
   ├─ Extract metadata
   │  ├─ File path
   │  ├─ Diff content
   │  ├─ Change type
   │  └─ Change statistics
   │
   ├─ Detect config type (yaml, json, xml, properties)
   │
   ├─ AI Analysis with ENHANCED prompts
   │  ├─ Environment context
   │  ├─ Change statistics
   │  ├─ File importance
   │  └─ Example output
   │
   └─ Store analysis result
   ↓
4. AI-Powered Overall Risk Assessment
   ├─ Aggregate individual file risks
   ├─ Calculate risk distribution
   ├─ Count policy violations
   ├─ AI holistic evaluation
   └─ Generate mitigation strategies
   ↓
5. Build enhanced analysis
   ├─ Original drift data
   ├─ Per-file AI analysis
   ├─ Overall risk assessment
   └─ Comprehensive recommendations
   ↓
6. Save to diff_analysis_*.json
   ↓
7. Return file path to Supervisor
```

---

## 🚀 Usage

### Via Supervisor (Recommended):

The Supervisor automatically calls this agent after Config Collector.

### Direct Usage:

```python
from Agents.workers.diff_policy_engine.diff_engine_agent import DiffPolicyEngineAgent
from shared.config import Config
from shared.models import TaskRequest

# Initialize
config = Config()
agent = DiffPolicyEngineAgent(config)

# Create task (requires drift_analysis file from Config Collector)
task = TaskRequest(
    task_id="test_analysis",
    task_type="analyze_drift",
    parameters={
        "drift_analysis_file": "config_data/drift_analysis/drift_analysis_20251004_165133.json"
    }
)

# Process task
result = agent.process_task(task)

print(f"Status: {result.status}")
print(f"Analysis file: {result.result['diff_analysis_file']}")
print(f"Files analyzed: {result.result['summary']['files_analyzed']}")
print(f"Overall risk: {result.result['summary']['overall_risk']}")
```

---

## 📁 Outputs

### Diff Analysis File
**Location:** `config_data/diff_analysis/diff_analysis_YYYYMMDD_HHMMSS.json`

**Structure:**
```json
{
  "drift_analysis": {
    "...": "original drift data from Config Collector"
  },
  "ai_policy_analysis": {
    "total_files_analyzed": 3,
    "policy_violations": [
      {
        "type": "security",
        "severity": "high",
        "description": "SSL disabled in production",
        "rule": "SEC_TLS_REQUIRED"
      }
    ],
    "overall_risk_level": "high",
    "risk_assessment": {
      "overall_risk_level": "high",
      "average_risk_score": 75,
      "risk_distribution": {
        "high": 2,
        "medium": 1,
        "low": 0
      },
      "risk_factors": [
        "2 high-risk changes in production",
        "1 security violation",
        "Critical environment - low risk tolerance"
      ],
      "mitigation_priority": "urgent",
      "mitigation_strategies": [
        "Revert SSL disablement immediately",
        "Review all security configurations",
        "Test changes in staging environment"
      ]
    },
    "recommendations": [
      "Immediate action required for security violations",
      "Comprehensive testing before deployment",
      "Review changes with security team"
    ]
  },
  "detailed_drifts_with_ai": [
    {
      "file_path": "config/application.yml",
      "has_changes": true,
      "diff_content": ["..."],
      "ai_analysis": {
        "risk_level": "high",
        "change_impact": "Disables SSL/TLS encryption",
        "policy_violations": [...],
        "recommendations": [...],
        "security_impact": "CRITICAL",
        "operational_impact": "HIGH"
      }
    }
  ]
}
```

---

## ✨ Enhanced AI Prompts

### What Makes Our Prompts Better?

#### 1. **Role Definition**
```
"You are an AI-powered configuration drift analysis assistant 
for the Golden Config AI system."
```

#### 2. **Rich Context**
```
### Context:
- Environment: production (Critical!)
- File: config/app.yml
- Configuration Type: yaml
- Change Summary:
  - Lines added: 5
  - Lines deleted: 3
  - Change type: modification
```

#### 3. **File Importance**
```
### File Importance:
This file impacts:
- Security (authentication, encryption, access control)
- Performance (resource limits, timeouts, caching)
- Compliance (regulatory requirements)
- Operational stability (availability, reliability)
```

#### 4. **Structured Task**
Clear enumeration of what AI needs to provide.

#### 5. **Example Output**
Shows AI exactly what format we expect.

### Result:
**More intelligent, context-aware, and actionable analysis!**

---

## 🎯 Risk Levels

### Critical
- Security controls disabled
- Authentication weakened
- Encryption removed
- Production data exposure

### High
- Major configuration changes
- Multiple policy violations
- Production environment
- Cascading failure risk

### Medium
- Standard configuration updates
- Minor policy concerns
- Staging environment
- Limited impact scope

### Low
- Minor changes
- Documentation updates
- Development environment
- No violations

---

## 🔧 Configuration

Set in `.env`:

```bash
# Worker Model (Claude 3 Haiku)
BEDROCK_WORKER_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# AWS Region
AWS_REGION=us-east-1
```

---

## 📊 Policy Checks

### Security Policies:
- SSL/TLS enforcement
- Authentication requirements
- Encryption standards
- Access control rules

### Compliance Policies:
- Data protection
- Audit logging
- Retention requirements
- Regulatory compliance

### Operational Policies:
- Resource limits
- Timeout configurations
- High availability settings
- Disaster recovery configs

---

## 🐛 Troubleshooting

### "Drift analysis file not found"
**Check:**
1. Config Collector completed successfully
2. File path is correct
3. File exists in `config_data/drift_analysis/`

### "Invalid JSON in drift analysis file"
**Check:**
1. Config Collector generated valid JSON
2. File is not corrupted
3. File encoding is UTF-8

### "AI analysis failed"
**Check:**
1. AWS Bedrock access is available
2. Model ID is correct
3. AWS credentials are valid
4. Diff content is not too large (>2000 chars truncated)

### "No drifts to analyze"
**Check:**
1. Config Collector found changes
2. `detailed_drifts` array is not empty
3. Files have `has_changes: true`

---

## 📚 Related

- **Supervisor:** `../../Supervisor/README.md`
- **Config Collector:** `../config_collector/README.md`
- **Main README:** `../../../README.md`

---

**The AI-powered analysis engine of your multi-agent system!** 🧠

