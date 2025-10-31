from hypothesis import given, strategies as st, settings, assume
import dask.array as da
import numpy as np

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

if __name__ == "__main__":
    test_eye_diagonal_ones()