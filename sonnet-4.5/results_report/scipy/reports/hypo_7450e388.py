import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings


@given(n=st.integers(min_value=1, max_value=10))
@settings(max_examples=20)
def test_inv_always_returns_sparse_matrix(n):
    A = sp.diags([2.0 + i for i in range(n)], format='csr')

    A_inv = spl.inv(A)

    assert sp.issparse(A_inv), f"inv() returned {type(A_inv)} instead of sparse matrix for {n}x{n} input"
    assert A_inv.shape == (n, n), f"inv() returned shape {A_inv.shape} instead of ({n}, {n})"


if __name__ == "__main__":
    test_inv_always_returns_sparse_matrix()