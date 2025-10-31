#!/usr/bin/env python3
"""Test to reproduce the reported bug"""

from hypothesis import given, assume, strategies as st
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb
import pytest

# First, test the property-based test from the bug report
@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6)
)
def test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected(vmin, vmax):
    assume(vmin > vmax)
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    with pytest.raises(ValueError):
        _rescale_imshow_rgb(darray, vmin=vmin, vmax=vmax, robust=False)

# Test with the specific failing input mentioned
def test_specific_case():
    print("Testing specific case: vmin=1.0, vmax=0.0")
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    try:
        result = _rescale_imshow_rgb(darray, vmin=1.0, vmax=0.0, robust=False)
        print(f"No error raised! Result shape: {result.shape}")
        print(f"Result min/max: {np.min(result)}, {np.max(result)}")
    except ValueError as e:
        print(f"ValueError raised: {e}")

# Also test the reproducing example from the bug report
def test_reproducing_example():
    print("\nTesting reproducing example from bug report:")
    darray = np.array([[[50.0, 50.0, 50.0]]]).astype('f8')

    print(f"Input array: {darray}")
    print(f"Input shape: {darray.shape}")

    try:
        result = _rescale_imshow_rgb(darray, vmin=100.0, vmax=0.0, robust=False)
        print(f"No error raised! Result: {result}")
        print(f"Result shape: {result.shape}")

        # Let's also check what happens with the formula
        manual_calc = (darray - 100.0) / (0.0 - 100.0)
        print(f"Manual calculation (darray - vmin) / (vmax - vmin): {manual_calc}")
        print(f"After clipping to [0,1]: {np.minimum(np.maximum(manual_calc, 0), 1)}")
    except ValueError as e:
        print(f"ValueError raised: {e}")

if __name__ == "__main__":
    # Run specific test cases
    test_specific_case()
    test_reproducing_example()

    # Run hypothesis test
    print("\nRunning hypothesis test...")
    try:
        test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected()
        print("Hypothesis test passed (all cases raised ValueError)")
    except AssertionError as e:
        print(f"Hypothesis test failed: Found case where ValueError was not raised")
        # Try to find a simple failing case
        print("\nTrying simple failing case: vmin=1.0, vmax=0.5")
        darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')
        try:
            result = _rescale_imshow_rgb(darray, vmin=1.0, vmax=0.5, robust=False)
            print("No error raised for vmin=1.0, vmax=0.5")
        except ValueError:
            print("Error correctly raised for vmin=1.0, vmax=0.5")