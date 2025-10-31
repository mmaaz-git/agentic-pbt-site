from hypothesis import given, strategies as st, assume
from scipy.cluster.vq import kmeans2
import numpy as np

@given(st.integers(min_value=5, max_value=30),
       st.integers(min_value=2, max_value=5),
       st.integers(min_value=2, max_value=8))
def test_kmeans2_different_minit_methods(n_obs, n_features, k):
    assume(k <= n_obs)
    data = np.random.randn(n_obs, n_features)
    centroids, labels = kmeans2(data, k, minit='random')
    assert centroids.shape == (k, n_features)
    assert len(labels) == n_obs

if __name__ == "__main__":
    test_kmeans2_different_minit_methods()