#!/usr/bin/env python3
"""Test to reproduce the scipy.signal.resample empty array bug"""

import numpy as np
import scipy.signal as signal
import traceback
from hypothesis import given, strategies as st, settings

# First, test the hypothesis test case
@settings(max_examples=100)
@given(
    data=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                  min_size=0, max_size=5)
)
def test_resample_empty_and_tiny_arrays(data):
    if len(data) == 0:
        x = np.array([], dtype=np.float64)
        y = signal.resample(x, 10)
        assert len(y) == 10

print("Testing with Hypothesis...")
try:
    test_resample_empty_and_tiny_arrays()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

# Now test the direct reproduction case
print("\nDirect reproduction test...")
try:
    x = np.array([], dtype=np.float64)
    print(f"Input array: {x}")
    print(f"Input shape: {x.shape}")
    y = signal.resample(x, 10)
    print(f"Output: {y}")
    print(f"Output shape: {y.shape}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test with a 2D empty array on different axes
print("\nTesting 2D empty array (axis=0)...")
try:
    x = np.array([[], []]).T  # Shape (0, 2)
    print(f"Input shape: {x.shape}")
    y = signal.resample(x, 10, axis=0)
    print(f"Output shape: {y.shape}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\nTesting 2D empty array (axis=1)...")
try:
    x = np.array([[], []])  # Shape (2, 0)
    print(f"Input shape: {x.shape}")
    y = signal.resample(x, 10, axis=1)
    print(f"Output shape: {y.shape}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Test with non-empty arrays to confirm normal behavior
print("\nTesting normal case with data...")
try:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"Input: {x}")
    y = signal.resample(x, 8)
    print(f"Output: {y}")
    print(f"Output shape: {y.shape}")
    print("Normal case works fine!")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")