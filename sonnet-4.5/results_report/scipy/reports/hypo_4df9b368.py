import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.cluster.vq import whiten

@given(st.integers(min_value=2, max_value=20),
       st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_whiten_zero_std_columns_unchanged(n_obs, n_features):
    rng = np.random.default_rng(42)
    obs = rng.standard_normal((n_obs, n_features))

    zero_col = rng.integers(0, n_features)
    constant_value = rng.uniform(-10, 10)
    obs[:, zero_col] = constant_value

    whitened = whiten(obs)

    assert np.allclose(whitened[:, zero_col], constant_value), \
        f"Column with zero std should remain unchanged. Got {whitened[:, zero_col]} instead of {constant_value}"

# Run the test
test_whiten_zero_std_columns_unchanged()