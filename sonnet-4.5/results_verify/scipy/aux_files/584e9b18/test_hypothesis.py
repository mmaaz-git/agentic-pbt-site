import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.sparse.csgraph import shortest_path


@st.composite
def positive_weighted_graphs(draw, max_size=10):
    n = draw(st.integers(min_value=2, max_value=max_size))
    matrix = draw(st.lists(
        st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
                 min_size=n, max_size=n),
        min_size=n, max_size=n
    ))
    matrix = np.array(matrix)
    np.fill_diagonal(matrix, 0)
    return matrix


@given(positive_weighted_graphs())
@settings(max_examples=100)
def test_direct_edge_not_longer_than_shortest_path(graph):
    dist_matrix = shortest_path(graph, directed=True)

    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] > 0:
                assert dist_matrix[i, j] <= graph[i, j] + 1e-9, \
                    f"Shortest path from {i} to {j} should not be longer than direct edge"

# Test with the specific failing input from the report
if __name__ == "__main__":
    print("Testing with the specific failing input from the bug report...")
    failing_graph = np.array([[0.00000000e+00, 1.69552992e-69], [0.00000000e+00, 0.00000000e+00]])
    print(f"Graph:\n{failing_graph}")

    dist_matrix = shortest_path(failing_graph, directed=True)
    print(f"Distance matrix:\n{dist_matrix}")

    # Check the assertion
    for i in range(failing_graph.shape[0]):
        for j in range(failing_graph.shape[1]):
            if failing_graph[i, j] > 0:
                print(f"Direct edge from {i} to {j}: {failing_graph[i, j]}")
                print(f"Shortest path from {i} to {j}: {dist_matrix[i, j]}")
                if dist_matrix[i, j] > failing_graph[i, j] + 1e-9:
                    print(f"ERROR: Shortest path is longer than direct edge!")
                else:
                    print(f"OK: Shortest path is not longer than direct edge")