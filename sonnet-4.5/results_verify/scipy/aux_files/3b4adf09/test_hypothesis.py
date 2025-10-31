from hypothesis import given, strategies as st, assume, settings
from scipy.cluster.vq import kmeans2
import numpy as np

@given(st.integers(min_value=5, max_value=30),
       st.integers(min_value=2, max_value=5),
       st.integers(min_value=2, max_value=8))
@settings(max_examples=50)  # Limit for faster execution
def test_kmeans2_different_minit_methods(n_obs, n_features, k):
    assume(k <= n_obs)
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(n_obs, n_features)

    print(f"Testing with n_obs={n_obs}, n_features={n_features}, k={k}")

    try:
        centroids, labels = kmeans2(data, k, minit='random')
        assert centroids.shape == (k, n_features)
        assert len(labels) == n_obs
        print(f"  Success!")
    except Exception as e:
        print(f"  Failed with {type(e).__name__}: {e}")
        if n_obs == n_features:
            print(f"  NOTE: This is a square matrix case (n_obs == n_features)")
        raise

# Run the test
if __name__ == "__main__":
    # Test the specific failing case mentioned in the report
    print("Testing the specific failing case: n_obs=5, n_features=5, k=2")

    # Test directly without the decorator
    n_obs, n_features, k = 5, 5, 2
    np.random.seed(42)
    data = np.random.randn(n_obs, n_features)
    print(f"Testing with n_obs={n_obs}, n_features={n_features}, k={k}")

    try:
        centroids, labels = kmeans2(data, k, minit='random')
        assert centroids.shape == (k, n_features)
        assert len(labels) == n_obs
        print(f"  Success!")
    except Exception as e:
        print(f"  Failed with {type(e).__name__}: {e}")
        if n_obs == n_features:
            print(f"  NOTE: This is a square matrix case (n_obs == n_features)")

    # Now run the hypothesis test
    print("\nRunning Hypothesis tests...")
    test_kmeans2_different_minit_methods()