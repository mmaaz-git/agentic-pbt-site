import numpy as np
from hypothesis import given, strategies as st, settings, assume


@given(
    st.floats(min_value=1e-100, max_value=1.0, allow_nan=False),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_geometric_always_positive(p, size):
    assume(0 < p <= 1.0)

    result = np.random.geometric(p, size)

    assert np.all(result >= 1), f"geometric({p}) returned invalid values: min={np.min(result)}"
    assert np.issubdtype(result.dtype, np.integer), f"geometric() should return integers"


if __name__ == "__main__":
    test_geometric_always_positive()