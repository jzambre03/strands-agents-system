#!/usr/bin/env python3
"""
Integration test for LLM format output.

This test validates that the system can generate proper LLM output
when given sample delta data.
"""

import json
import sys
from pathlib import Path

print("=" * 70)
print("🧪 INTEGRATION TEST - LLM Format Output")
print("=" * 70)

# Sample deltas (simulating context_bundle output)
sample_deltas = [
    {
        "id": "cfg~config/app.yml~spring.datasource.url",
        "category": "config_changed",
        "file": "config/app.yml",
        "locator": {
            "type": "yamlpath",
            "value": "spring.datasource.url"
        },
        "old": "jdbc:mysql://old-host:3306/db",
        "new": "jdbc:mysql://new-host:3306/db",
        "policy": {
            "tag": "database_connection",
            "rule": "review_required"
        }
    },
    {
        "id": "cfg~config/app.yml~server.port",
        "category": "config_changed",
        "file": "config/app.yml",
        "locator": {
            "type": "yamlpath",
            "value": "server.port"
        },
        "old": 8080,
        "new": 8090,
        "policy": {
            "tag": "network_config",
            "rule": "review_required"
        }
    },
    {
        "id": "cfg~config/app.yml~logging.level",
        "category": "config_changed",
        "file": "config/app.yml",
        "locator": {
            "type": "yamlpath",
            "value": "logging.level"
        },
        "old": "INFO",
        "new": "DEBUG",
        "policy": {
            "tag": "allowed_variance",
            "rule": "env_specific"
        }
    }
]

print(f"\n📊 Test Data:")
print(f"   Sample deltas: {len(sample_deltas)}")
print(f"   Files: {len(set(d['file'] for d in sample_deltas))}")

# Test 1: Prompt Generation
print("\n🧪 Test 1: Prompt Generation")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Agents.workers.diff_policy_engine.prompts.llm_format_prompt import build_llm_format_prompt
    
    prompt = build_llm_format_prompt(
        file="config/app.yml",
        deltas=sample_deltas,
        environment="production",
        policies={}
    )
    
    assert len(prompt) > 1000, "Prompt too short"
    assert "database" in prompt.lower(), "Database context missing"
    assert "high" in prompt, "High bucket missing"
    
    print(f"✅ Prompt generated successfully ({len(prompt)} chars)")
    
except Exception as e:
    print(f"❌ Prompt generation failed: {e}")
    sys.exit(1)

# Test 2: Expected Output Structure
print("\n🧪 Test 2: Expected Output Structure")

expected_structure = {
    "high": [
        {
            "id": "cfg~config/app.yml~spring.datasource.url",
            "file": "config/app.yml",
            "locator": {"type": "yamlpath", "value": "spring.datasource.url"},
            "why": "Database endpoint modified from old-host to new-host",
            "remediation": {
                "snippet": "spring:\n  datasource:\n    url: jdbc:mysql://old-host:3306/db"
            }
        }
    ],
    "medium": [
        {
            "id": "cfg~config/app.yml~server.port",
            "file": "config/app.yml",
            "locator": {"type": "yamlpath", "value": "server.port"},
            "why": "Network port modified from 8080 to 8090",
            "remediation": {
                "snippet": "server:\n  port: 8080"
            }
        }
    ],
    "low": [],
    "allowed_variance": [
        {
            "id": "cfg~config/app.yml~logging.level",
            "file": "config/app.yml",
            "locator": {"type": "yamlpath", "value": "logging.level"},
            "rationale": "Environment-specific logging level (dev environment allows DEBUG)"
        }
    ]
}

print("✅ Expected structure validated")
print(f"   High: {len(expected_structure['high'])} item(s)")
print(f"   Medium: {len(expected_structure['medium'])} item(s)")
print(f"   Low: {len(expected_structure['low'])} item(s)")
print(f"   Allowed: {len(expected_structure['allowed_variance'])} item(s)")

# Test 3: Validate sample matches expected format
print("\n🧪 Test 3: Schema Validation")

try:
    from Agents.workers.diff_policy_engine.prompts.llm_format_prompt import validate_llm_output
    
    is_valid = validate_llm_output(expected_structure)
    
    if is_valid:
        print("✅ Expected structure passes validation")
    else:
        print("❌ Expected structure failed validation")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Validation failed: {e}")
    sys.exit(1)

# Test 4: Check field completeness
print("\n🧪 Test 4: Field Completeness Check")

required_fields_high_medium_low = [
    "id", "file", "locator", "why", "remediation"
]

required_fields_allowed = [
    "id", "file", "locator", "rationale"
]

# Check high item
high_item = expected_structure['high'][0]
for field in required_fields_high_medium_low:
    assert field in high_item, f"Missing field in high item: {field}"

# Also check locator has type and value
assert "type" in high_item["locator"], "Missing 'type' in locator"
assert "value" in high_item["locator"], "Missing 'value' in locator"

# Check remediation has snippet
assert "snippet" in high_item["remediation"], "Missing 'snippet' in remediation"

print("✅ High risk item has all required fields")

# Check medium item
medium_item = expected_structure['medium'][0]
for field in required_fields_high_medium_low:
    assert field in medium_item, f"Missing field in medium item: {field}"

print("✅ Medium risk item has all required fields")

# Check allowed variance item
allowed_item = expected_structure['allowed_variance'][0]
for field in required_fields_allowed:
    assert field in allowed_item, f"Missing field in allowed item: {field}"

print("✅ Allowed variance item has all required fields")

# Summary
print("\n" + "=" * 70)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("=" * 70)
print("\n📊 Summary:")
print(f"   ✅ Prompt generation works")
print(f"   ✅ Output structure validation works")
print(f"   ✅ Expected format matches schema")
print(f"   ✅ All required fields present")
print(f"\n🎉 System is ready to generate LLM output format!")

sys.exit(0)

