import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from hypothesis import given, strategies as st, assume
from xarray.compat.dask_array_compat import reshape_blockwise

@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=20)
)
def test_reshape_blockwise_shape_correct(rows, cols):
    total_size = rows * cols
    arr = da.arange(total_size).reshape(rows, cols)
    new_shape = (total_size,)
    reshaped = reshape_blockwise(arr, new_shape)
    assert reshaped.shape == new_shape, f"Shape should match: {reshaped.shape} vs {new_shape}"

if __name__ == "__main__":
    # Test the specific failing case manually
    rows, cols = 1, 1
    total_size = rows * cols
    arr = da.arange(total_size).reshape(rows, cols)
    new_shape = (total_size,)
    reshaped = reshape_blockwise(arr, new_shape)
    assert reshaped.shape == new_shape, f"Shape should match: {reshaped.shape} vs {new_shape}"
    print("Test passed!")