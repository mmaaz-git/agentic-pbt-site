#!/usr/bin/env python3
"""Test the hypothesis test case from the bug report"""

from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import integrate

@given(
    k=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    x_min=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    x_max=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10)  # Reduce for quick testing
def test_tanhsinh_constant(k, x_min, x_max):
    assume(x_min < x_max)
    print(f"Testing: k={k}, x_min={x_min}, x_max={x_max}")
    result = integrate.tanhsinh(lambda x: k, x_min, x_max)
    expected = k * (x_max - x_min)
    assert np.isclose(result.integral, expected, rtol=1e-10)
    print(f"  Success: result={result.integral}, expected={expected}")

# Test the specific failing case mentioned
print("Testing specific failing case: k=0.0, x_min=0.0, x_max=1.0")
try:
    test_tanhsinh_constant(k=0.0, x_min=0.0, x_max=1.0)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Run the hypothesis test
print("\nRunning hypothesis tests...")
try:
    test_tanhsinh_constant()
except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}: {e}")