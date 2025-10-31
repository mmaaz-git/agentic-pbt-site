import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

@st.composite
def csr_matrices(draw):
    n = draw(st.integers(min_value=2, max_value=20))
    m = draw(st.integers(min_value=2, max_value=20))
    density = draw(st.floats(min_value=0.1, max_value=0.4))
    return sp.random(n, m, density=density, format='csr')

@given(csr_matrices())
@settings(max_examples=50)
def test_sorted_indices_flag_invalidation(A):
    A.sort_indices()
    assert A.has_sorted_indices

    if A.nnz >= 2:
        A.indices[0], A.indices[1] = A.indices[1], A.indices[0]

        indices_actually_sorted = all(
            np.all(A.indices[A.indptr[i]:A.indptr[i+1]][:-1] <=
                   A.indices[A.indptr[i]:A.indptr[i+1]][1:])
            for i in range(A.shape[0]) if A.indptr[i+1] - A.indptr[i] > 1
        )

        if A.has_sorted_indices and not indices_actually_sorted:
            raise AssertionError(f"BUG: has_sorted_indices flag not invalidated after indices modification\n"
                               f"Matrix shape: {A.shape}, nnz: {A.nnz}\n"
                               f"has_sorted_indices: {A.has_sorted_indices}\n"
                               f"Indices after swap: {A.indices[:10]}...")

if __name__ == "__main__":
    test_sorted_indices_flag_invalidation()