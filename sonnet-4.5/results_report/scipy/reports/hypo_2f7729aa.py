import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.cluster.vq import kmeans2
import pytest


def fixed_size_arrays(min_rows, max_rows, num_cols, min_val=-1e6, max_val=1e6):
    return st.builds(
        np.array,
        st.lists(
            st.lists(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False),
                    min_size=num_cols, max_size=num_cols),
            min_size=min_rows, max_size=max_rows
        )
    )


@given(obs=fixed_size_arrays(10, 50, 3, -100, 100),
       k=st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_kmeans2_cholesky_crash_on_degenerate_data(obs, k):
    assume(len(obs) >= k)

    obs_degenerate = np.zeros_like(obs)
    obs_degenerate[:, :] = obs[:, 0:1]

    try:
        codebook, labels = kmeans2(obs_degenerate, k, iter=1, minit='random')
        assert True
    except np.linalg.LinAlgError as e:
        if "not positive definite" in str(e):
            pytest.fail(f"kmeans2 with minit='random' crashes on rank-deficient data: {e}")
        raise

if __name__ == "__main__":
    test_kmeans2_cholesky_crash_on_degenerate_data()