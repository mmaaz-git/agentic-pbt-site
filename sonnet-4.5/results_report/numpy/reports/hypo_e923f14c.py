import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(1, 100),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000)
    )
)
@settings(max_examples=2000)
def test_mean_respects_bounds(arr):
    mean_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    assert min_val <= mean_val <= max_val, f"Mean {mean_val} not in bounds [{min_val}, {max_val}]"


if __name__ == "__main__":
    test_mean_respects_bounds()