import numpy as np
from scipy.cluster.vq import kmeans2
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst


@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50),
        elements=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
    ),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_kmeans2_handles_low_variance_data(obs, k):
    assume(obs.shape[0] > k)
    assume(np.std(obs) > 1e-6)

    try:
        centroids, labels = kmeans2(obs, k, iter=5, missing='raise')
        assert centroids.shape[0] <= k
        assert centroids.shape[1] == obs.shape[1]
    except Exception as e:
        if "empty" in str(e).lower():
            pass
        else:
            raise

if __name__ == "__main__":
    test_kmeans2_handles_low_variance_data()