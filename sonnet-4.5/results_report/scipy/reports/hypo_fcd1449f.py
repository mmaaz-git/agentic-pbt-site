import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.spatial import ConvexHull

@settings(max_examples=500)
@given(
    n_points=st.integers(min_value=3, max_value=20),
    n_dims=st.integers(min_value=2, max_value=3),
    data=st.data()
)
def test_convexhull_incremental_equals_batch(n_points, n_dims, data):
    points_list = data.draw(st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=n_dims, max_size=n_dims),
        min_size=n_points, max_size=n_points
    ))
    points = np.array(points_list)

    new_point_list = data.draw(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=n_dims, max_size=n_dims))
    new_point = np.array([new_point_list])

    try:
        hull_incremental = ConvexHull(points, incremental=True)
        hull_incremental.add_points(new_point)

        all_points = np.vstack([points, new_point])
        hull_batch = ConvexHull(all_points)

        assert np.isclose(hull_incremental.volume, hull_batch.volume), \
            f"Incremental and batch ConvexHull should have same volume: {hull_incremental.volume} vs {hull_batch.volume}"
    except Exception as e:
        if "QhullError" in str(type(e).__name__):
            assume(False)
        raise

# Run the test
if __name__ == "__main__":
    test_convexhull_incremental_equals_batch()