#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, assume
import scipy.special as sp
import numpy as np

@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=1, max_value=50),
    st.floats(min_value=0.01, max_value=0.99)
)
@settings(max_examples=500)
def test_bdtri_round_trip(k, n, p):
    # Note: The test includes assume(k <= n) to avoid the failing cases
    assume(k <= n)

    y = sp.bdtr(k, n, p)
    p_reconstructed = sp.bdtri(k, n, y)
    y_reconstructed = sp.bdtr(k, n, p_reconstructed)

    assert np.isclose(y, y_reconstructed, rtol=1e-8, atol=1e-10)

# Run with the constraint
print("Running hypothesis test WITH k <= n constraint:")
try:
    test_bdtri_round_trip()
    print("Test PASSED with k <= n constraint")
except Exception as e:
    print(f"Test FAILED with k <= n constraint: {e}")

# Now let's test without the constraint to see failures
@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=1, max_value=50),
    st.floats(min_value=0.01, max_value=0.99)
)
@settings(max_examples=100)
def test_bdtri_round_trip_no_constraint(k, n, p):
    # No assume statement - allow k >= n cases

    y = sp.bdtr(k, n, p)
    p_reconstructed = sp.bdtri(k, n, y)
    y_reconstructed = sp.bdtr(k, n, p_reconstructed)

    if k >= n:
        # We expect this to fail for k >= n
        if not np.isnan(p_reconstructed):
            print(f"Unexpected: bdtri({k}, {n}, {y}) = {p_reconstructed} (not NaN)")

    assert np.isclose(y, y_reconstructed, rtol=1e-8, atol=1e-10), \
        f"Failed for k={k}, n={n}, p={p}: y={y}, p_reconstructed={p_reconstructed}, y_reconstructed={y_reconstructed}"

print("\nRunning hypothesis test WITHOUT k <= n constraint (expecting failures):")
try:
    test_bdtri_round_trip_no_constraint()
    print("Test PASSED without k <= n constraint (unexpected!)")
except Exception as e:
    print(f"Test FAILED without k <= n constraint (expected): {str(e)[:200]}...")