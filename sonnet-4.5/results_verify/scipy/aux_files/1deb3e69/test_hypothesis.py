from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.cluster.hierarchy as hierarchy

@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=50),
        elements=st.floats(
            min_value=-1e6, max_value=1e6,
            allow_nan=False, allow_infinity=False
        )
    ),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=200)
def test_fcluster_maxclust_count(obs, n_clusters):
    assume(obs.shape[0] >= n_clusters)
    assume(obs.shape[1] > 0)

    try:
        Z = hierarchy.linkage(obs, method='ward')
        clusters = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')

        unique_clusters = len(np.unique(clusters))
        assert unique_clusters == n_clusters, \
            f"Expected {n_clusters} clusters, got {unique_clusters}"
    except Exception as e:
        if "Must have n>=2 objects" in str(e):
            assume(False)
        raise

# Test with the specific failing case
print("Testing with specific failing input:")
obs = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
n_clusters = 2

try:
    test_fcluster_maxclust_count(obs, n_clusters)
    print("Test passed")
except AssertionError as e:
    print(f"Test failed: {e}")

# Run property-based test
print("\nRunning property-based tests...")
test_fcluster_maxclust_count()
print("Property-based tests completed")