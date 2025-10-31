from hypothesis import given, assume, strategies as st
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb
import pytest

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6)
)
def test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected(vmin, vmax):
    assume(vmin > vmax)
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    with pytest.raises(ValueError):
        _rescale_imshow_rgb(darray, vmin=vmin, vmax=vmax, robust=False)

# Run the test
if __name__ == "__main__":
    test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected()