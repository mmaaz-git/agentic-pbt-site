#!/usr/bin/env python3
"""Run the hypothesis tests from the bug report"""

import scipy.datasets
import os
import shutil
import platformdirs

# Ensure cache directory doesn't exist for testing
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

def test_clear_cache_rejects_non_callables():
    """clear_cache should reject non-callable inputs regardless of cache state"""
    invalid_input = "test_string"
    if not callable(invalid_input):
        try:
            scipy.datasets.clear_cache(invalid_input)
            return "FAILED: Function completed without raising an error"
        except (AssertionError, TypeError) as e:
            return f"PASSED: Raised {type(e).__name__}: {e}"
        except Exception as e:
            return f"FAILED: Raised unexpected {type(e).__name__}: {e}"


def test_clear_cache_rejects_invalid_callables():
    """clear_cache should validate callable names regardless of cache state"""
    def invalid_dataset():
        pass

    try:
        scipy.datasets.clear_cache(invalid_dataset)
        return "FAILED: Function completed without raising an error"
    except ValueError as e:
        if "doesn't exist" in str(e):
            return f"PASSED: Raised ValueError: {e}"
        return f"FAILED: ValueError with wrong message: {e}"
    except Exception as e:
        return f"FAILED: Raised unexpected {type(e).__name__}: {e}"

print("Testing with cache directory NOT existing:")
print("=" * 60)
print("\nTest 1: Non-callable input")
print(test_clear_cache_rejects_non_callables())

print("\nTest 2: Invalid callable")
print(test_clear_cache_rejects_invalid_callables())

# Now test with cache existing
print("\n\nTesting with cache directory existing:")
print("=" * 60)
os.makedirs(cache_dir, exist_ok=True)

print("\nTest 3: Non-callable input (cache exists)")
print(test_clear_cache_rejects_non_callables())

print("\nTest 4: Invalid callable (cache exists)")
print(test_clear_cache_rejects_invalid_callables())

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)