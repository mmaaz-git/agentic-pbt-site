from hypothesis import given, settings, strategies as st
import scipy.sparse as sp

@given(
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=2, max_value=6)
)
@settings(max_examples=30)
def test_kron_with_identity(m, n):
    """kron(A, I) should only store nonzero elements"""
    A = sp.random(m, n, density=0.5)
    I = sp.eye(m)

    result = sp.kron(A, I)

    # The number of nonzeros should be m times A's nonzeros
    # Each nonzero in A contributes m nonzeros from the diagonal of I
    assert result.nnz == m * A.nnz

# Run the test
test_kron_with_identity()