import numpy as np
from hypothesis import given, strategies as st, assume, settings, seed
from scipy.spatial import ConvexHull

# Run a limited number of tests to find failures
@settings(max_examples=50, deadline=None)
@seed(42)
@given(
    n_points=st.integers(min_value=3, max_value=10),
    n_dims=st.integers(min_value=2, max_value=3),
    data=st.data()
)
def test_convexhull_incremental_equals_batch(n_points, n_dims, data):
    points_list = data.draw(st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10),
                 min_size=n_dims, max_size=n_dims),
        min_size=n_points, max_size=n_points
    ))
    points = np.array(points_list)

    new_point_list = data.draw(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                                   min_value=-10, max_value=10),
                                        min_size=n_dims, max_size=n_dims))
    new_point = np.array([new_point_list])

    try:
        hull_incremental = ConvexHull(points, incremental=True)
        hull_incremental.add_points(new_point)

        all_points = np.vstack([points, new_point])
        hull_batch = ConvexHull(all_points)

        if not np.isclose(hull_incremental.volume, hull_batch.volume, rtol=1e-9, atol=1e-12):
            print(f"FAILURE FOUND:")
            print(f"  Initial points shape: {points.shape}")
            print(f"  New point: {new_point.tolist()}")
            print(f"  Incremental volume: {hull_incremental.volume}")
            print(f"  Batch volume: {hull_batch.volume}")
            print(f"  Difference: {abs(hull_incremental.volume - hull_batch.volume)}")
            return False
    except Exception as e:
        if "QhullError" in str(type(e).__name__):
            assume(False)
        raise

    return True

# Run the test
print("Running hypothesis tests...")
failures = 0
for i in range(50):
    try:
        if not test_convexhull_incremental_equals_batch():
            failures += 1
            if failures >= 5:  # Stop after finding 5 failures
                break
    except:
        pass

print(f"\nTotal failures found: {failures}")