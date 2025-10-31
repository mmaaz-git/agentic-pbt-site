#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import given, settings, strategies as st
import pytest
import scipy.datasets
import tempfile
import os
import shutil
from scipy.datasets._utils import _clear_cache

# First, let's test if the hypothesis test itself fails
@settings(max_examples=10)
@given(st.sampled_from(["invalid", 123, 45.6, {"key": "value"}]))
def test_clear_cache_rejects_invalid_inputs(invalid_input):
    """Original test from bug report - testing public API"""
    # Temporarily override cache directory to ensure it doesn't exist
    temp_cache = tempfile.mkdtemp()
    non_existent_cache = os.path.join(temp_cache, "scipy-data-test")

    try:
        # Test with non-existent cache
        with pytest.raises((ValueError, TypeError, AssertionError)):
            _clear_cache(invalid_input, cache_dir=non_existent_cache)
    finally:
        shutil.rmtree(temp_cache)

# Run the test
print("Running hypothesis test...")
try:
    test_clear_cache_rejects_invalid_inputs()
    print("All hypothesis tests passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")