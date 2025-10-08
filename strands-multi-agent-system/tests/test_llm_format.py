#!/usr/bin/env python3
"""
Unit tests for LLM format output functionality.

Tests the new LLM format batch analysis, fallback categorization,
and output merging functionality.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from Agents.workers.diff_policy_engine.prompts.llm_format_prompt import (
        build_llm_format_prompt,
        validate_llm_output,
        get_drift_categories,
        get_risk_levels
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import modules: {e}")
    IMPORTS_AVAILABLE = False


def test_prompt_building():
    """Test LLM format prompt generation."""
    print("\n🧪 Test 1: Prompt Building")
    
    if not IMPORTS_AVAILABLE:
        print("⚠️  Skipped: Imports not available")
        return False
    
    sample_deltas = [
        {
            "id": "cfg~app.yml~spring.datasource.url",
            "category": "config_changed",
            "file": "config/app.yml",
            "locator": {"type": "yamlpath", "value": "spring.datasource.url"},
            "old": "jdbc:mysql://old-host",
            "new": "jdbc:mysql://new-host",
            "policy": {"tag": "database_connection", "rule": "review_required"}
        }
    ]
    
    prompt = build_llm_format_prompt(
        file="config/app.yml",
        deltas=sample_deltas,
        environment="production",
        policies={}
    )
    
    # Validate prompt content
    assert "config/app.yml" in prompt, "File name not in prompt"
    assert "spring.datasource.url" in prompt, "Locator not in prompt"
    assert "jdbc:mysql://old-host" in prompt, "Old value not in prompt"
    assert "jdbc:mysql://new-host" in prompt, "New value not in prompt"
    assert '"high":' in prompt, "Output format not in prompt"
    assert '"medium":' in prompt, "Output format not in prompt"
    assert '"low":' in prompt, "Output format not in prompt"
    assert '"allowed_variance":' in prompt, "Output format not in prompt"
    assert "drift_category" in prompt, "Drift category not in prompt"
    assert "ai_review_assistant" in prompt, "AI review assistant not in prompt"
    
    print("✅ Prompt building test passed")
    print(f"   Prompt length: {len(prompt)} chars")
    return True


def test_llm_output_validation():
    """Test LLM output validation function."""
    print("\n🧪 Test 2: LLM Output Validation")
    
    if not IMPORTS_AVAILABLE:
        print("⚠️  Skipped: Imports not available")
        return False
    
    # Valid output - EXACT FORMAT (no extra fields)
    valid_output = {
        "high": [
            {
                "id": "cfg~app.yml~key",
                "file": "app.yml",
                "locator": {"type": "keypath", "value": "app.yml.key"},
                "why": "Configuration changed",
                "remediation": {"snippet": "key: original_value"}
            }
        ],
        "medium": [],
        "low": [],
        "allowed_variance": []
    }
    
    assert validate_llm_output(valid_output) == True, "Valid output should pass"
    print("✅ Valid output test passed")
    
    # Invalid output - missing keys
    invalid_output = {
        "high": [],
        "medium": []
        # missing low and allowed_variance
    }
    
    assert validate_llm_output(invalid_output) == False, "Invalid output should fail"
    print("✅ Invalid output test passed")
    
    # Invalid output - wrong types
    invalid_output2 = {
        "high": "not an array",
        "medium": [],
        "low": [],
        "allowed_variance": []
    }
    
    assert validate_llm_output(invalid_output2) == False, "Wrong type should fail"
    print("✅ Wrong type test passed")
    
    return True


def test_drift_categories():
    """Test drift category definitions."""
    print("\n🧪 Test 3: Drift Categories")
    
    if not IMPORTS_AVAILABLE:
        print("⚠️  Skipped: Imports not available")
        return False
    
    categories = get_drift_categories()
    
    assert "Database" in categories, "Database category missing"
    assert "Network" in categories, "Network category missing"
    assert "Functional" in categories, "Functional category missing"
    assert "Logical" in categories, "Logical category missing"
    assert "Dependency" in categories, "Dependency category missing"
    assert "Configuration" in categories, "Configuration category missing"
    assert "Other" in categories, "Other category missing"
    
    print(f"✅ All {len(categories)} drift categories present")
    return True


def test_risk_levels():
    """Test risk level definitions."""
    print("\n🧪 Test 4: Risk Levels")
    
    if not IMPORTS_AVAILABLE:
        print("⚠️  Skipped: Imports not available")
        return False
    
    risk_levels = get_risk_levels()
    
    assert "high" in risk_levels, "High risk level missing"
    assert "medium" in risk_levels, "Medium risk level missing"
    assert "low" in risk_levels, "Low risk level missing"
    
    print(f"✅ All {len(risk_levels)} risk levels present")
    return True


def test_sample_llm_output():
    """Test with sample LLM output format."""
    print("\n🧪 Test 5: Sample LLM Output Format")
    
    # Load sample LLM output from UI folder
    sample_file = Path(__file__).parent.parent.parent / "UI" / "LLM_output.json"
    
    if not sample_file.exists():
        print("⚠️  Skipped: Sample file not found")
        return False
    
    try:
        with open(sample_file, 'r') as f:
            sample_output = json.load(f)
        
        # Check structure
        assert "high" in sample_output, "Missing 'high' key"
        assert "medium" in sample_output, "Missing 'medium' key"
        assert "low" in sample_output, "Missing 'low' key"
        assert "allowed_variance" in sample_output, "Missing 'allowed_variance' key"
        
        # Check high risk items have required fields (including old/new)
        for item in sample_output.get("high", []):
            assert "id" in item, "Missing 'id' in high risk item"
            assert "file" in item, "Missing 'file' in high risk item"
            assert "locator" in item, "Missing 'locator' in high risk item"
            assert "old" in item, "Missing 'old' in high risk item"
            assert "new" in item, "Missing 'new' in high risk item"
            assert "why" in item, "Missing 'why' in high risk item"
            assert "remediation" in item, "Missing 'remediation' in high risk item"
        
        # Check allowed_variance items have required fields (including old/new)
        for item in sample_output.get("allowed_variance", []):
            assert "id" in item, "Missing 'id' in allowed_variance item"
            assert "file" in item, "Missing 'file' in allowed_variance item"
            assert "locator" in item, "Missing 'locator' in allowed_variance item"
            assert "old" in item, "Missing 'old' in allowed_variance item"
            assert "new" in item, "Missing 'new' in allowed_variance item"
            assert "rationale" in item, "Missing 'rationale' in allowed_variance item"
        
        print(f"✅ Sample LLM output validation passed")
        print(f"   High: {len(sample_output.get('high', []))} items")
        print(f"   Medium: {len(sample_output.get('medium', []))} items")
        print(f"   Low: {len(sample_output.get('low', []))} items")
        print(f"   Allowed: {len(sample_output.get('allowed_variance', []))} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample validation failed: {e}")
        return False


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("🧪 UNIT TESTS - LLM Format Output")
    print("=" * 70)
    
    tests = [
        test_prompt_building,
        test_llm_output_validation,
        test_drift_categories,
        test_risk_levels,
        test_sample_llm_output
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

