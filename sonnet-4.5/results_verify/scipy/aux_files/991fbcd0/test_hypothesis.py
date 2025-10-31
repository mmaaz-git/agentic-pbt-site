import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings, seed

@st.composite
def csr_matrices(draw):
    n = draw(st.integers(min_value=2, max_value=20))
    m = draw(st.integers(min_value=2, max_value=20))
    density = draw(st.floats(min_value=0.1, max_value=0.4))
    return sp.random(n, m, density=density, format='csr')

@given(csr_matrices())
@settings(max_examples=10)  # Reduced for testing
@seed(12345)  # For reproducibility
def test_sorted_indices_flag_invalidation(A):
    A.sort_indices()
    assert A.has_sorted_indices

    if A.nnz >= 2:
        # Swap first two indices
        A.indices[0], A.indices[1] = A.indices[1], A.indices[0]

        # Check if indices are actually sorted
        indices_actually_sorted = all(
            np.all(A.indices[A.indptr[i]:A.indptr[i+1]][:-1] <=
                   A.indices[A.indptr[i]:A.indptr[i+1]][1:])
            for i in range(A.shape[0]) if A.indptr[i+1] - A.indptr[i] > 1
        )

        if A.has_sorted_indices and not indices_actually_sorted:
            raise AssertionError("BUG: has_sorted_indices flag not invalidated after indices modification")

# Run the test
print("Running property-based test...")
try:
    test_sorted_indices_flag_invalidation()
    print("Test completed without finding the bug (shouldn't happen)")
except AssertionError as e:
    print(f"BUG FOUND: {e}")
    print("The test correctly identified the bug!")