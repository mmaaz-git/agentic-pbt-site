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
    # Test with the specific failing input
    p = 1e-100
    np.random.seed(42)
    try:
        result = np.random.geometric(p, size=10)
        print(f"Testing with p={p}")
        print(f"Results: {result}")
        print(f"Min value: {np.min(result)}")
        print(f"All positive: {np.all(result >= 1)}")
    except Exception as e:
        print(f"Error: {e}")