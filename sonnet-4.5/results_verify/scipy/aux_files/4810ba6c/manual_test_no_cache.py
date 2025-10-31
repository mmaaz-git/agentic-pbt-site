import scipy.datasets
import os
import shutil

# Ensure cache directory doesn't exist
cache_dir = os.path.expanduser("~/.cache/scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed existing cache directory: {cache_dir}")

print("\nTesting with cache directory NOT existing...")

print("\n1. Testing with invalid string:")
try:
    scipy.datasets.clear_cache("invalid_string")
    print("   No error raised - BUG CONFIRMED!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

print("\n2. Testing with integer:")
try:
    scipy.datasets.clear_cache(42)
    print("   No error raised - BUG CONFIRMED!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

print("\n3. Testing with dict:")
try:
    scipy.datasets.clear_cache({"key": "value"})
    print("   No error raised - BUG CONFIRMED!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

print("\n4. Testing with valid None:")
try:
    scipy.datasets.clear_cache(None)
    print("   Successfully handled None with no cache")
except Exception as e:
    print(f"   Unexpected error with None: {e}")