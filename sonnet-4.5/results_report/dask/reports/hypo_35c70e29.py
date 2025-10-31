import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=3),
    st.integers(min_value=0, max_value=5)
)
@settings(max_examples=200)
def test_squeeze_raises_correct_error_for_invalid_axis(shape, axis):
    if axis < len(shape):
        return

    x_np = np.random.rand(*shape)
    x_da = da.from_array(x_np, chunks='auto')

    np_error = None
    da_error = None

    try:
        np.squeeze(x_np, axis=axis)
    except Exception as e:
        np_error = type(e).__name__

    try:
        da.squeeze(x_da, axis=axis).compute()
    except Exception as e:
        da_error = type(e).__name__

    if np_error is not None and da_error is not None:
        assert np_error == da_error or (np_error == 'AxisError' and da_error in ['AxisError', 'ValueError']), \
            f"NumPy raises {np_error}, but Dask raises {da_error} for shape={shape}, axis={axis}"

if __name__ == "__main__":
    test_squeeze_raises_correct_error_for_invalid_axis()