import numpy as np
from hypothesis import given, settings, strategies as st
from xarray.core import duck_array_ops


@given(
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=2, max_value=8)
)
@settings(max_examples=100)
def test_cumprod_axis_none_matches_numpy(rows, cols):
    values = np.random.randn(rows, cols)

    xr_result = duck_array_ops.cumprod(values, axis=None)
    np_result = np.cumprod(values, axis=None)

    assert xr_result.shape == np_result.shape, \
        f"Shape mismatch: {xr_result.shape} != {np_result.shape}"
    assert np.allclose(xr_result.flatten(), np_result), \
        f"Values mismatch"


@given(
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=2, max_value=8)
)
@settings(max_examples=100)
def test_cumsum_axis_none_matches_numpy(rows, cols):
    values = np.random.randn(rows, cols)

    xr_result = duck_array_ops.cumsum(values, axis=None)
    np_result = np.cumsum(values, axis=None)

    assert xr_result.shape == np_result.shape, \
        f"Shape mismatch: {xr_result.shape} != {np_result.shape}"
    assert np.allclose(xr_result.flatten(), np_result), \
        f"Values mismatch"


# Run tests
if __name__ == "__main__":
    print("Testing cumprod...")
    test_cumprod_axis_none_matches_numpy()
    print("Testing cumsum...")
    test_cumsum_axis_none_matches_numpy()