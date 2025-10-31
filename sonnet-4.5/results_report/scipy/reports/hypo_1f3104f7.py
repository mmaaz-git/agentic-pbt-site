import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.sparse.csgraph import csgraph_from_masked, csgraph_to_masked


@st.composite
def graph_matrices(draw, max_size=20):
    n = draw(st.integers(min_value=2, max_value=max_size))
    matrix = draw(st.lists(
        st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
                 min_size=n, max_size=n),
        min_size=n, max_size=n
    ))
    return np.array(matrix)


@given(graph_matrices())
@settings(max_examples=200)
def test_csgraph_masked_roundtrip(graph):
    masked_graph = np.ma.masked_equal(graph, 0)
    sparse_graph = csgraph_from_masked(masked_graph)
    reconstructed = csgraph_to_masked(sparse_graph)

    assert np.allclose(masked_graph.data, reconstructed.data, equal_nan=True)
    assert np.array_equal(masked_graph.mask, reconstructed.mask)


if __name__ == "__main__":
    test_csgraph_masked_roundtrip()