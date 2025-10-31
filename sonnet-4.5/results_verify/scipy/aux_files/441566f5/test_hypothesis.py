import numpy as np
import scipy.spatial
import scipy.spatial.distance as dist
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

@given(
    x=arrays(dtype=np.float64, shape=(5,), elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
    y=arrays(dtype=np.float64, shape=(5,), elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
)
@settings(max_examples=1000)
def test_minkowski_distance_consistency_with_euclidean(x, y):
    d_minkowski = scipy.spatial.minkowski_distance(x, y, p=2)
    d_euclidean = dist.euclidean(x, y)
    assert np.isclose(d_minkowski, d_euclidean, rtol=1e-10, atol=0)

# Run the test to see if it fails
if __name__ == "__main__":
    test_minkowski_distance_consistency_with_euclidean()
    print("All tests passed!")