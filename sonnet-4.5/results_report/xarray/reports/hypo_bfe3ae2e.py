import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.plot.utils import _rescale_imshow_rgb


@given(
    arr=st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
)
@settings(max_examples=1000)
def test_rescale_imshow_rgb_with_robust(arr):
    darray = np.array(arr)
    result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
    assert np.all(result >= 0), f"Result has values < 0: {result}"
    assert np.all(result <= 1), f"Result has values > 1: {result}"

if __name__ == "__main__":
    test_rescale_imshow_rgb_with_robust()