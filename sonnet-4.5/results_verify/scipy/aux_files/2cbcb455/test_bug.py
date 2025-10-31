#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

import tempfile
import os
import shutil
import sys
from scipy.datasets._utils import _clear_cache
import scipy.datasets

print("=" * 60)
print("Test 1: Reproducing the bug with non-existent cache directory")
print("=" * 60)

# Create a temporary directory setup
temp_cache = tempfile.mkdtemp()
non_existent_cache = os.path.join(temp_cache, "scipy-data-test")

print(f"Cache exists: {os.path.exists(non_existent_cache)}")

try:
    _clear_cache("invalid_string", cache_dir=non_existent_cache)
    print("BUG CONFIRMED: No exception raised for invalid input!")
    bug_confirmed = True
except (ValueError, TypeError, AssertionError) as e:
    print(f"Exception raised as expected: {type(e).__name__}: {e}")
    bug_confirmed = False

# Clean up
shutil.rmtree(temp_cache)

print("\n" + "=" * 60)
print("Test 2: Testing with various invalid inputs")
print("=" * 60)

invalid_inputs = ["invalid", 123, 45.6, {"key": "value"}]

for invalid_input in invalid_inputs:
    temp_cache = tempfile.mkdtemp()
    non_existent_cache = os.path.join(temp_cache, "scipy-data-test")

    print(f"\nTesting with input: {invalid_input} (type: {type(invalid_input).__name__})")
    try:
        _clear_cache(invalid_input, cache_dir=non_existent_cache)
        print(f"  -> No exception raised (BUG)")
    except (ValueError, TypeError, AssertionError) as e:
        print(f"  -> Exception raised: {type(e).__name__}: {e}")

    shutil.rmtree(temp_cache)

print("\n" + "=" * 60)
print("Test 3: Testing behavior when cache directory EXISTS")
print("=" * 60)

# Test with existing cache directory
temp_cache = tempfile.mkdtemp()
cache_dir = os.path.join(temp_cache, "scipy-data-test")
os.makedirs(cache_dir)

print(f"Cache exists: {os.path.exists(cache_dir)}")

for invalid_input in invalid_inputs:
    print(f"\nTesting with input: {invalid_input} (type: {type(invalid_input).__name__})")
    try:
        _clear_cache(invalid_input, cache_dir=cache_dir)
        print(f"  -> No exception raised (UNEXPECTED)")
    except (ValueError, TypeError, AssertionError) as e:
        print(f"  -> Exception raised: {type(e).__name__}: {e}")

shutil.rmtree(temp_cache)

print("\n" + "=" * 60)
print("Test 4: Testing the public API function")
print("=" * 60)

# Test the public API as well
for invalid_input in invalid_inputs:
    print(f"\nTesting scipy.datasets.clear_cache with: {invalid_input}")
    try:
        # We need to ensure cache doesn't exist for this test
        import platformdirs
        cache_dir = platformdirs.user_cache_dir("scipy-data")
        if os.path.exists(cache_dir):
            print(f"  -> Skipping test (cache exists at {cache_dir})")
            continue

        scipy.datasets.clear_cache(invalid_input)
        print(f"  -> No exception raised (BUG)")
    except (ValueError, TypeError, AssertionError) as e:
        print(f"  -> Exception raised: {type(e).__name__}: {e}")
    except ImportError:
        print("  -> Skipping (platformdirs not available)")