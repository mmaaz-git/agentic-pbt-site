#!/usr/bin/env python3
"""
Minimal reproduction of scipy.datasets.clear_cache bug.
This demonstrates that invalid inputs are silently accepted when cache doesn't exist.
"""

import scipy.datasets
import os
import shutil

# Ensure cache directory doesn't exist for consistent reproduction
cache_dir = os.path.expanduser("~/.cache/scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed existing cache directory: {cache_dir}")

print("Testing with cache directory NOT existing:")
print("-" * 50)

# Test with invalid string input
print("\n1. Testing with string input 'invalid_string':")
try:
    scipy.datasets.clear_cache("invalid_string")
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {e}")

# Test with invalid integer input
print("\n2. Testing with integer input 42:")
try:
    scipy.datasets.clear_cache(42)
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {e}")

# Test with invalid dict input
print("\n3. Testing with dict input {'key': 'value'}:")
try:
    scipy.datasets.clear_cache({"key": "value"})
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {e}")

print("\n" + "=" * 50)
print("\nNow creating cache directory and testing again...")
print("=" * 50)

# Create the cache directory
os.makedirs(cache_dir, exist_ok=True)
print(f"Created cache directory: {cache_dir}")

print("\nTesting with cache directory EXISTING:")
print("-" * 50)

# Test with invalid string input (should fail now)
print("\n1. Testing with string input 'invalid_string':")
try:
    scipy.datasets.clear_cache("invalid_string")
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

# Test with invalid integer input (should fail now)
print("\n2. Testing with integer input 42:")
try:
    scipy.datasets.clear_cache(42)
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

# Test with invalid dict input (should fail now)
print("\n3. Testing with dict input {'key': 'value'}:")
try:
    scipy.datasets.clear_cache({"key": "value"})
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"\nCleaned up cache directory: {cache_dir}")