import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph


@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=2**31 - 1),
    st.booleans()
)
@settings(max_examples=200)
def test_laplacian_form_consistency(n, max_weight, seed, normed):
    np.random.seed(seed)

    dense = np.random.rand(n, n) * max_weight
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0)

    graph = csr_matrix(dense)

    lap_array = csgraph.laplacian(graph, normed=normed, form='array')
    lap_lo = csgraph.laplacian(graph, normed=normed, form='lo')

    test_vec = np.random.rand(n)

    result_array = lap_array @ test_vec
    result_lo = lap_lo @ test_vec

    assert np.allclose(result_array, result_lo, rtol=1e-9, atol=1e-9)

# Run the test
if __name__ == "__main__":
    test_laplacian_form_consistency()