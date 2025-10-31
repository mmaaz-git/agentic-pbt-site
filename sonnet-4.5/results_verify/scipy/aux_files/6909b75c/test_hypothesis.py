#!/usr/bin/env python3
"""Test the hypothesis test case"""

from hypothesis import given, strategies as st, settings
import numpy as np
from scipy import stats

@given(
    data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100
    )
)
@settings(max_examples=10)  # Limit examples for quick testing
def test_quantile_zero_is_min(data):
    x = np.array(data)
    q0 = stats.quantile(x, 0)  # This will fail with integer 0
    min_x = np.min(x)
    assert np.allclose(q0, min_x), f"quantile(x, 0) should be min(x)"

if __name__ == "__main__":
    print("Running hypothesis test with integer 0...")
    try:
        test_quantile_zero_is_min()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")

    # Now test with explicit float
    print("\nTesting manually with float 0.0...")
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = np.array(test_data)
    try:
        q0 = stats.quantile(x, 0.0)  # Using float
        min_x = np.min(x)
        assert np.allclose(q0, min_x)
        print(f"Manual test passed: quantile(x, 0.0) = {q0}, min(x) = {min_x}")
    except Exception as e:
        print(f"Manual test failed: {type(e).__name__}: {e}")