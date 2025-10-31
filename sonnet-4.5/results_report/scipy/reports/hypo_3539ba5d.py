from hypothesis import given, strategies as st
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

@given(st.integers(min_value=3, max_value=12))
def test_floyd_warshall_overwrite_consistency(n):
    graph = np.random.rand(n, n) * 10
    graph_c = np.ascontiguousarray(graph, dtype=np.float64)
    graph_f = np.asfortranarray(graph, dtype=np.float64)

    dist_c = floyd_warshall(graph_c, directed=True, overwrite=False)
    dist_f = floyd_warshall(graph_f, directed=True, overwrite=True)

    np.testing.assert_allclose(dist_c, dist_f, rtol=1e-10, atol=1e-10)

test_floyd_warshall_overwrite_consistency()