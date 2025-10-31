import io
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=20))
@settings(max_examples=500)
def test_roundtrip_1d_int_array_row_orientation(lst):
    original = {'x': np.array(lst)}

    f = io.BytesIO()
    savemat(f, original, oned_as='row')
    f.seek(0)
    loaded = loadmat(f)

    np_arr = np.array(lst)
    if np_arr.ndim == 1 and np_arr.size > 0:
        expected_shape = (1, len(lst))
    elif np_arr.ndim == 1 and np_arr.size == 0:
        expected_shape = (1, 0)
    else:
        expected_shape = np_arr.shape

    assert loaded['x'].shape == expected_shape, f"Expected shape {expected_shape} but got {loaded['x'].shape} for lst={lst}"

if __name__ == "__main__":
    test_roundtrip_1d_int_array_row_orientation()
    print("All tests passed!")