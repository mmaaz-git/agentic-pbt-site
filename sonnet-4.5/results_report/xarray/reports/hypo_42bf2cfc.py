import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.plot.utils import _rescale_imshow_rgb

@given(st.lists(st.floats(min_value=0, max_value=255, allow_nan=False, allow_infinity=False), min_size=10, max_size=100))
@settings(max_examples=500)
def test_rescale_imshow_rgb_robust(values):
    darray = np.array(values)
    result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
    assert np.all(result >= 0)
    assert np.all(result <= 1)

if __name__ == "__main__":
    test_rescale_imshow_rgb_robust()