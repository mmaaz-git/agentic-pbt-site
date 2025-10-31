import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.spatial import Delaunay


@given(
    st.integers(min_value=10, max_value=30),
    st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=500)
def test_delaunay_point_location(n_points, seed):
    np.random.seed(seed)
    points = np.random.randn(n_points, 2)

    try:
        tri = Delaunay(points)

        for i in range(len(points)):
            simplex_idx = tri.find_simplex(points[i])

            assert simplex_idx >= 0, \
                f"Input point {i} at {points[i]} not found in any simplex"

    except Exception as e:
        if "degenerate" in str(e).lower():
            assume(False)
        raise

if __name__ == "__main__":
    # Run the test
    try:
        test_delaunay_point_location()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")