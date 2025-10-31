#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, assume
import dask.array as da

# Copy the test from the bug report
@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=-1, max_value=1)
)
@settings(max_examples=300, deadline=None)
def test_eye_diagonal_ones(N, M, k):
    """
    Property: eye creates identity matrix with ones on diagonal
    Evidence: eye creates matrix with 1s on main diagonal
    """
    assume(N > abs(k) and M > abs(k))

    arr = da.eye(N, chunks=3, M=M, k=k)
    computed = arr.compute()

    for i in range(N):
        for j in range(M):
            if j - i == k:
                assert computed[i, j] == 1.0
            else:
                assert computed[i, j] == 0.0

print("Running hypothesis test...")
print("This will test many combinations of N, M, k values")
print("-" * 50)

try:
    test_eye_diagonal_ones()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")
    print("\nTrying specific failing case: N=2, M=3, k=0")
    try:
        test_eye_diagonal_ones(2, 3, 0)
    except Exception as e2:
        print(f"Confirmed failure: {e2}")