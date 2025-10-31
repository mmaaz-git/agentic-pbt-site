import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon


@settings(max_examples=500)
@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_jensenshannon_base_produces_finite_result(k, base):
    x = np.random.rand(k) + 0.1
    y = np.random.rand(k) + 0.1
    x = x / x.sum()
    y = y / y.sum()

    result = jensenshannon(x, y, base=base)

    assert np.isfinite(result), \
        f"Jensen-Shannon should produce finite result for base={base}, got {result}"

if __name__ == "__main__":
    test_jensenshannon_base_produces_finite_result()