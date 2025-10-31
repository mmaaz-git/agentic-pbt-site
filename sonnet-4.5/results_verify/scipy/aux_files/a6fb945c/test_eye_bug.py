from hypothesis import given, strategies as st, settings
import scipy.sparse as sp

@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=-30, max_value=30)
)
@settings(max_examples=200)
def test_eye_with_large_offset(n, k):
    """eye should handle all offsets consistently"""
    E = sp.eye_array(n, k=k, format='csr')

    if abs(k) >= n:
        assert E.nnz == 0
    else:
        expected_nnz = n - abs(k)
        assert E.nnz == expected_nnz

if __name__ == "__main__":
    test_eye_with_large_offset()