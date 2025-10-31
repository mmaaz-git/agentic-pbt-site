#!/usr/bin/env python3
"""Run the hypothesis tests from the bug report"""

from hypothesis import given, strategies as st, settings
from xarray.core.utils import is_uniform_spaced
import numpy as np
import traceback

# Test 1: linspace should always be uniform
@given(n=st.integers(min_value=0, max_value=100))
@settings(max_examples=300)
def test_linspace_always_uniform(n):
    arr = np.linspace(0, 10, n)
    result = is_uniform_spaced(arr)
    assert result == True, f"linspace with {n} points should be uniformly spaced"

# Test 2: small arrays shouldn't crash
@given(size=st.integers(min_value=0, max_value=2))
@settings(max_examples=100)
def test_small_arrays_dont_crash(size):
    arr = list(range(size))
    result = is_uniform_spaced(arr)
    assert isinstance(result, bool)

print("Running test_linspace_always_uniform...")
try:
    test_linspace_always_uniform()
    print("  No failures found!")
except Exception as e:
    print(f"  Failed with: {e}")
    print("  This test should fail on n=0 or n=1")

print("\nRunning test_small_arrays_dont_crash...")
try:
    test_small_arrays_dont_crash()
    print("  No failures found!")
except Exception as e:
    print(f"  Failed with: {e}")
    print("  This test should fail on size=0 or size=1")

# Show specific failing cases
print("\n" + "="*50)
print("Specific failing cases:")

for n in [0, 1, 2, 3]:
    print(f"\nnp.linspace(0, 10, {n}):")
    arr = np.linspace(0, 10, n)
    print(f"  Array: {arr}")
    print(f"  Length: {len(arr)}")
    try:
        result = is_uniform_spaced(arr)
        print(f"  is_uniform_spaced result: {result}")
    except ValueError as e:
        print(f"  ValueError: {e}")