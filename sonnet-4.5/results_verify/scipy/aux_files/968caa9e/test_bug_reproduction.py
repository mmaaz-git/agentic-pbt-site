#!/usr/bin/env python3
"""Test script to reproduce the bug in scipy.datasets.clear_cache"""

import scipy.datasets
import os
import shutil
import platformdirs

# First, ensure cache directory doesn't exist
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    print(f"Removing existing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)

print("=" * 60)
print("Testing scipy.datasets.clear_cache with invalid inputs")
print("=" * 60)

# Test 1: Non-callable input
print("\nTest 1: Passing a non-callable string 'not_callable'")
print("-" * 40)
try:
    scipy.datasets.clear_cache("not_callable")
    print("Result: Function completed without raising an error")
except AssertionError as e:
    print(f"Result: AssertionError raised: {e}")
except TypeError as e:
    print(f"Result: TypeError raised: {e}")
except Exception as e:
    print(f"Result: Other exception raised: {type(e).__name__}: {e}")

# Test 2: Invalid callable (not a dataset method)
print("\nTest 2: Passing an invalid callable")
print("-" * 40)
def invalid_dataset():
    pass

try:
    scipy.datasets.clear_cache(invalid_dataset)
    print("Result: Function completed without raising an error")
except ValueError as e:
    print(f"Result: ValueError raised: {e}")
except AssertionError as e:
    print(f"Result: AssertionError raised: {e}")
except Exception as e:
    print(f"Result: Other exception raised: {type(e).__name__}: {e}")

# Test 3: For comparison - test with cache directory existing
print("\n" + "=" * 60)
print("Testing with cache directory existing")
print("=" * 60)

# Create cache directory
os.makedirs(cache_dir, exist_ok=True)
print(f"Created cache directory: {cache_dir}")

print("\nTest 3: Passing a non-callable string 'not_callable' (cache exists)")
print("-" * 40)
try:
    scipy.datasets.clear_cache("not_callable")
    print("Result: Function completed without raising an error")
except AssertionError as e:
    print(f"Result: AssertionError raised: {e}")
except TypeError as e:
    print(f"Result: TypeError raised: {e}")
except Exception as e:
    print(f"Result: Other exception raised: {type(e).__name__}: {e}")

print("\nTest 4: Passing an invalid callable (cache exists)")
print("-" * 40)
try:
    scipy.datasets.clear_cache(invalid_dataset)
    print("Result: Function completed without raising an error")
except ValueError as e:
    print(f"Result: ValueError raised: {e}")
except AssertionError as e:
    print(f"Result: AssertionError raised: {e}")
except Exception as e:
    print(f"Result: Other exception raised: {type(e).__name__}: {e}")

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"\nCleaned up cache directory: {cache_dir}")