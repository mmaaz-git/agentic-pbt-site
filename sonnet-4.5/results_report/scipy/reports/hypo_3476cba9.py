import numpy as np
from hypothesis import assume, given, settings, strategies as st, example
from scipy.spatial import Delaunay


@given(
    st.integers(min_value=4, max_value=25),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=500)
@example(8, 4)  # The specific failing case
def test_delaunay_vertices_found_by_find_simplex(n_points, n_dims):
    np.random.seed(0)  # Fix seed for reproducibility
    points = np.random.randn(n_points, n_dims) * 100

    try:
        tri = Delaunay(points)
    except Exception:
        assume(False)

    simplex_indices_default = tri.find_simplex(points)
    simplex_indices_bruteforce = tri.find_simplex(points, bruteforce=True)

    failed_default = simplex_indices_default < 0
    failed_bruteforce = simplex_indices_bruteforce < 0

    assert np.all(~failed_bruteforce), "All vertices should be found by bruteforce"
    assert np.all(~failed_default), f"All vertices should be found by default algorithm, but {np.sum(failed_default)} were not"


if __name__ == "__main__":
    # Run the test - will fail on the example(8, 4) case
    test_delaunay_vertices_found_by_find_simplex()