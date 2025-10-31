from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp
from hypothesis import given, strategies as st


@st.composite
def simple_graphs(draw):
    n = draw(st.integers(min_value=2, max_value=5))
    graph = csr_matrix((n, n), dtype=float)

    for i in range(n-1):
        graph[i, i+1] = 1.0
        graph[i+1, i] = 1.0

    return graph


@given(simple_graphs())
def test_laplacian_form_array_returns_numpy_array(graph):
    lap = laplacian(graph, normed=False, form='array')

    assert not sp.issparse(lap), \
        f"laplacian(form='array') should return numpy array, got {type(lap).__name__}"

if __name__ == "__main__":
    test_laplacian_form_array_returns_numpy_array()