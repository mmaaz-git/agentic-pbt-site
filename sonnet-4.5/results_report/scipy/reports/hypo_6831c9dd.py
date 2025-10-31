import numpy as np
import scipy.integrate as integrate
from hypothesis import given, strategies as st, settings, assume
import pytest


@given(
    n_points=st.integers(min_value=4, max_value=10),
    dup_idx=st.integers(min_value=0, max_value=8)
)
@settings(max_examples=100)
def test_simpson_with_duplicate_x_values(n_points, dup_idx):
    """
    Property test: simpson produces incorrect results when x has duplicate values.

    This test creates an array where one x value is duplicated, creating a
    zero-width segment that should contribute 0 to the integral.
    """
    assume(dup_idx < n_points - 1)

    x = np.linspace(0, 1, n_points)
    x[dup_idx + 1] = x[dup_idx]
    y = x.copy()

    result = integrate.simpson(y, x=x)
    expected = 0.5

    if not np.isclose(result, expected, rtol=0.01):
        pytest.fail(f"simpson gives wrong result with duplicate x: {result} != {expected}")


if __name__ == "__main__":
    # Run the test manually to find a failing case
    test_simpson_with_duplicate_x_values()