import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.cluster.vq import kmeans2


@given(st.integers(min_value=2, max_value=20),
       st.integers(min_value=2, max_value=10))
@settings(max_examples=100)
def test_kmeans2_random_init_handles_rank_deficient_data(n_obs, n_features):
    data = np.random.randn(n_obs, 1)
    data = np.tile(data, (1, n_features))

    centroid, label = kmeans2(data, min(2, n_obs), minit='random')

    assert centroid.shape[1] == n_features

# Run the test
if __name__ == "__main__":
    test_kmeans2_random_init_handles_rank_deficient_data()