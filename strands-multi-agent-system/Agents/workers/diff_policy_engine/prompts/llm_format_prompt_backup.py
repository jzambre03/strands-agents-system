"""
LLM Format Prompt Template for Direct Output Generation

This module provides prompt templates that instruct the AI to return
the exact LLM output format (high/medium/low/allowed_variance) directly,
eliminating the need for post-processing transformations.

Based on adjudicator.md specification.
"""

from typing import List, Dict, Any


def build_llm_format_prompt(
    file: str,
    deltas: List[Dict[str, Any]],
    environment: str = "production",
    policies: Dict[str, Any] = None
) -> str:
    """
    Build an AI prompt that returns LLM output format directly.
    
    The prompt instructs the AI to analyze configuration deltas and return
    a JSON object with high/medium/low/allowed_variance arrays, where each
    item includes all required fields for the UI.
    
    Args:
        file: File path being analyzed
        deltas: List of delta objects from context_bundle
        environment: Target environment (production, staging, dev, qa)
        policies: Policy rules and guidelines
    
    Returns:
        Complete prompt string for AI analysis
    
    Output Format:
        {
          "high": [...],
          "medium": [...],
          "low": [...],
          "allowed_variance": [...]
        }
    """
    if policies is None:
        policies = {}
    
    # Build deltas summary with clear numbering
    deltas_summary = []
    for idx, delta in enumerate(deltas, 1):
        locator = delta.get('locator', {})
        deltas_summary.append({
            "index": idx,
            "delta_id": delta.get('id', 'unknown'),
            "locator_type": locator.get('type', 'unknown'),
            "locator_value": locator.get('value', 'unknown'),
            "old_value": str(delta.get('old')) if delta.get('old') is not None else "null",
            "new_value": str(delta.get('new')) if delta.get('new') is not None else "null",
            "policy_tag": delta.get('policy', {}).get('tag', 'unknown'),
            "policy_rule": delta.get('policy', {}).get('rule', 'none'),
            "category": delta.get('category', 'unknown')
        })
    
    # Build the comprehensive prompt
    prompt = f"""You are a configuration drift adjudicator analyzing file "{file}" for environment "{environment}".

Your task is to categorize ALL {len(deltas)} configuration changes into risk buckets and provide detailed analysis for each.

## CONTEXT
- **File**: {file}
- **Environment**: {environment}
- **Total Changes**: {len(deltas)} deltas
- **Analysis Required**: Every delta must be categorized in exactly ONE bucket

## POLICY GUIDELINES

### Security Policies:
- **BLOCK**: Credential changes (passwords, secrets, keys, tokens)
- **BLOCK**: Authentication/authorization configuration changes
- **BLOCK**: Security bypass or disabled security features
- **WARN**: Unencrypted credentials or sensitive data exposure

### Operational Policies:
- **WARN**: Database connection changes (JDBC URLs, hosts, ports)
- **WARN**: Network endpoint changes (URLs, ports, timeouts)
- **WARN**: Performance-critical settings (memory, threads, pools)
- **WARN**: Feature flags that alter user-visible behavior

### Compliance Policies:
- **REVIEW**: Configuration standard violations
- **REVIEW**: Logging/monitoring setting changes
- **REVIEW**: Audit trail modifications

### Allowed Variance:
- Environment-specific settings (dev/qa/staging differences)
- Test-only configuration
- Build/CI pipeline settings
- Documentation and comments
- Version metadata

## CHANGES TO ANALYZE

"""
    
    # Add each delta with clear formatting
    for d in deltas_summary:
        prompt += f"""
### CHANGE #{d['index']}
- **ID**: `{d['delta_id']}`
- **Category**: {d['category']}
- **Location**: {d['locator_type']}: `{d['locator_value']}`
- **Old Value**: `{d['old_value']}`
- **New Value**: `{d['new_value']}`
- **Policy Tag**: {d['policy_tag']}
- **Policy Rule**: {d['policy_rule']}

"""

    # Add output format specification
    prompt += f"""
## OUTPUT FORMAT

Return ONLY valid JSON with this EXACT structure (no other text before or after):

```json
{{
  "high": [
    {{
      "id": "delta_id_here",
      "file": "{file}",
      "locator": {{
        "type": "yamlpath|jsonpath|keypath|unidiff|coord|path",
        "value": "full.path.to.key"
      }},
      "old": "old_value",
      "new": "new_value",
      "drift_category": "Database|Network|Functional|Logical|Dependency|Configuration|Other",
      "risk_level": "high",
      "risk_reason": "One sentence explaining why this is high risk",
      "why": "What changed and its impact",
      "remediation": {{
        "snippet": "corrected configuration here"
      }},
      "ai_review_assistant": {{
        "potential_risk": "Specific risk description",
        "suggested_action": "Concrete action to take"
      }}
    }}
  ],
  "medium": [
    {{
      "id": "...",
      "file": "{file}",
      "locator": {{}},
      "old": "...",
      "new": "...",
      "drift_category": "...",
      "risk_level": "medium",
      "risk_reason": "Why medium risk",
      "why": "What changed",
      "remediation": {{
        "snippet": "..."
      }},
      "ai_review_assistant": {{
        "potential_risk": "...",
        "suggested_action": "..."
      }}
    }}
  ],
  "low": [
    {{
      "id": "...",
      "file": "{file}",
      "locator": {{}},
      "old": "...",
      "new": "...",
      "drift_category": "...",
      "risk_level": "low",
      "risk_reason": "Why low risk",
      "why": "What changed",
      "remediation": {{
        "snippet": "..."
      }},
      "ai_review_assistant": {{
        "potential_risk": "...",
        "suggested_action": "..."
      }}
    }}
  ],
  "allowed_variance": [
    {{
      "id": "...",
      "file": "{file}",
      "locator": {{}},
      "old": "...",
      "new": "...",
      "drift_category": "...",
      "why_allowed": "Policy rule or rationale for allowing this change",
      "risk_level": "low",
      "risk_reason": "Why this is low risk despite being allowed",
      "ai_review_assistant": {{
        "potential_risk": "Minimal risk explanation",
        "suggested_action": "Document and monitor"
      }}
    }}
  ]
}}
```

## DRIFT CATEGORY DEFINITIONS

Choose ONE category that best describes the change:

- **Database**: JDBC URLs, database drivers, connection strings, usernames, passwords, pool settings, schema names, table references
- **Network**: HTTP/HTTPS endpoints, API URLs, service hosts, ports, proxy settings, timeouts, circuit breakers, retry policies
- **Functional**: User-visible features, business logic, new API endpoints, feature flags, behavior changes
- **Logical**: Boolean conditions, null checks, validation rules, exception handling, control flow, comparisons
- **Dependency**: Library versions, Maven/NPM packages, build tool dependencies, plugin versions
- **Configuration**: Application settings, Spring profiles, environment variables, logging levels, cache settings, container configuration
- **Other**: Binary files, archives, metadata, documentation, unrecognized changes

## RISK LEVEL GUIDELINES

### **high** (Immediate attention required):
- Credentials or secrets changed
- Security features disabled or bypassed
- Production database connections modified
- Breaking changes to critical paths
- Authentication/authorization changes

### **medium** (Review recommended):
- Network endpoint changes
- Feature behavior modifications
- Performance setting adjustments
- Non-breaking API changes
- Timeout or retry policy changes

### **low** (Minor review):
- Logging level changes
- Comment or documentation updates
- Minor configuration tweaks
- Test-only changes
- Build metadata

## ALLOWED VARIANCE CRITERIA

Place in `allowed_variance` bucket if:
- Environment-specific configuration (dev vs qa vs prod differences are normal)
- Test suite configuration
- Build/CI pipeline settings that don't affect runtime
- Documentation or code comments only
- Version/build metadata
- Policy explicitly allows this via `policy.tag == "allowed_variance"`

## CRITICAL RULES

1. **JSON Only**: Return ONLY valid JSON, no explanatory text before or after
2. **All Deltas**: Every one of the {len(deltas)} deltas MUST appear in exactly ONE bucket
3. **Use Actual Values**: Copy the exact delta IDs, old/new values, and locators from the changes above
4. **Be Specific**: Provide concrete, actionable explanations in `why`, `risk_reason`, and `remediation`
5. **Actionable Remediation**: The `snippet` should be copy-pasteable corrected configuration
6. **Complete Analysis**: Every item must have ALL required fields filled in
7. **No Placeholders**: No "...", "TBD", or "N/A" - provide real content
8. **Order Matters**: Sort items within each bucket by delta ID for consistency

## EXAMPLE ITEMS

### High Risk Example:
```json
{{
  "id": "cfg~config/app.yml.datasource.password",
  "file": "config/app.yml",
  "locator": {{"type": "yamlpath", "value": "datasource.password"}},
  "old": "ENC(old_secret)",
  "new": "ENC(new_secret)",
  "drift_category": "Database",
  "risk_level": "high",
  "risk_reason": "Production database password changed without authorization",
  "why": "Database credential modified, could cause authentication failures",
  "remediation": {{
    "snippet": "datasource:\\n  password: ENC(old_secret)"
  }},
  "ai_review_assistant": {{
    "potential_risk": "Service outage if database rejects new credentials",
    "suggested_action": "Verify password change was approved and test connectivity before deployment"
  }}
}}
```

### Allowed Variance Example:
```json
{{
  "id": "cfg~k8s/deployment.yml.probe.timeout",
  "file": "k8s/deployment.yml",
  "locator": {{"type": "yamlpath", "value": "spec.containers[0].readinessProbe.timeoutSeconds"}},
  "old": 5,
  "new": 10,
  "drift_category": "Configuration",
  "why_allowed": "Environment-specific timing adjustment for dev environment",
  "risk_level": "low",
  "risk_reason": "Dev environment probe tuning, not production-facing",
  "ai_review_assistant": {{
    "potential_risk": "None for dev; would need review if promoted to production",
    "suggested_action": "Document timing variance and block promotion without SRE approval"
  }}
}}
```

## YOUR ANALYSIS

Analyze ALL {len(deltas)} changes above and categorize them. Return ONLY the JSON object with your complete analysis.

Remember:
- Be thorough and specific
- Use exact values from the deltas
- Provide actionable remediation
- Every delta must be categorized
- Return ONLY valid JSON
"""
    
    return prompt


def get_drift_categories() -> List[str]:
    """Get list of valid drift categories."""
    return [
        "Database",
        "Network",
        "Functional",
        "Logical",
        "Dependency",
        "Configuration",
        "Other"
    ]


def get_risk_levels() -> List[str]:
    """Get list of valid risk levels."""
    return ["high", "medium", "low"]


def validate_llm_output(output: Dict[str, Any]) -> bool:
    """
    Validate that LLM output has required structure.
    
    Args:
        output: LLM response to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["high", "medium", "low", "allowed_variance"]
    
    # Check top-level keys
    if not all(key in output for key in required_keys):
        return False
    
    # Check each bucket is a list
    for key in required_keys:
        if not isinstance(output[key], list):
            return False
    
    # Check items have required fields
    item_required_fields = ["id", "file", "locator", "old", "new"]
    
    for bucket in required_keys:
        for item in output[bucket]:
            if not isinstance(item, dict):
                return False
                
            if not all(field in item for field in item_required_fields):
                return False
            
            # Check bucket-specific requirements
            if bucket != "allowed_variance":
                if "drift_category" not in item:
                    return False
                if "risk_level" not in item:
                    return False
                # Note: "why" and "remediation" are recommended but not strictly required
            else:
                # allowed_variance items should have why_allowed
                if "why_allowed" not in item:
                    return False
    
    return True




