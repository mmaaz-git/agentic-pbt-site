from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse.csgraph as csg

@settings(max_examples=500)
@given(st.integers(min_value=2, max_value=10).flatmap(
    lambda n: st.tuples(
        st.just(n),
        st.lists(st.lists(st.floats(min_value=0, max_value=100,
                                    allow_nan=False, allow_infinity=False),
                         min_size=n, max_size=n),
                min_size=n, max_size=n)
    )
))
def test_csgraph_round_trip_dense(args):
    n, graph_list = args
    graph = np.array(graph_list, dtype=float)

    csgraph_sparse = csg.csgraph_from_dense(graph, null_value=0)
    graph_reconstructed = csg.csgraph_to_dense(csgraph_sparse, null_value=0)

    assert graph_reconstructed.shape == graph.shape
    np.testing.assert_array_equal(graph, graph_reconstructed)

if __name__ == "__main__":
    test_csgraph_round_trip_dense()