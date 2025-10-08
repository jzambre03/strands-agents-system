"""
Drift Analyzer Module
Provides precise configuration drift analysis with line-level locators.
Based on context-generator drift.py.
"""

from .drift import (
    # Core analysis functions
    extract_repo_tree,
    classify_files,
    diff_structural,
    semantic_config_diff,
    extract_dependencies,
    dependency_diff,
    
    # Specialized detectors
    detector_spring_profiles,
    detector_jenkinsfile,
    build_code_hunk_deltas,
    build_binary_deltas,
    
    # Bundle generation
    emit_context_bundle,
)

__all__ = [
    'extract_repo_tree',
    'classify_files',
    'diff_structural',
    'semantic_config_diff',
    'extract_dependencies',
    'dependency_diff',
    'detector_spring_profiles',
    'detector_jenkinsfile',
    'build_code_hunk_deltas',
    'build_binary_deltas',
    'emit_context_bundle',
]

