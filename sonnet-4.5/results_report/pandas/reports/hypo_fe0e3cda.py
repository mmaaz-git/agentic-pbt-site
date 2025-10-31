from hypothesis import given, strategies as st
import numpy as np
from pandas._libs.internals import BlockPlacement
from pandas.core.internals.api import maybe_infer_ndim

@given(st.integers(min_value=0, max_value=5))
def test_maybe_infer_ndim_returns_only_1_or_2(ndim):
    """Property: maybe_infer_ndim should only return 1 or 2 for numpy arrays"""
    shape = tuple([2] * max(ndim, 1)) if ndim > 0 else ()
    if ndim == 0:
        arr = np.array(5)
    else:
        arr = np.arange(np.prod(shape)).reshape(shape)

    placement = BlockPlacement(slice(0, 1))
    result = maybe_infer_ndim(arr, placement, ndim=None)

    assert result in [1, 2], f"Expected 1 or 2, got {result} for array with ndim={arr.ndim}"

if __name__ == "__main__":
    test_maybe_infer_ndim_returns_only_1_or_2()