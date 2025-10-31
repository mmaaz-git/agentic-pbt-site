#!/usr/bin/env python3
"""Reproduce the is_uniform_spaced bug"""

from xarray.core.utils import is_uniform_spaced
import traceback

# Test 1: Empty array
print("Test 1: Empty array")
try:
    result = is_uniform_spaced([])
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 2: Single element array
print("Test 2: Single element array")
try:
    result = is_uniform_spaced([5])
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 3: Two element array (should work)
print("Test 3: Two element array")
try:
    result = is_uniform_spaced([1, 2])
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 4: Normal array
print("Test 4: Normal array")
try:
    result = is_uniform_spaced([1, 2, 3, 4, 5])
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")