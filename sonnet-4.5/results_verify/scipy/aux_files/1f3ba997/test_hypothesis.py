from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.io.matlab as matlab
import tempfile
import os

@given(
    npst.arrays(
        dtype=npst.floating_dtypes(),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)
    )
)
@settings(max_examples=100)
def test_roundtrip_1d_arrays(arr):
    assume(not np.any(np.isnan(arr)) and not np.any(np.isinf(arr)))

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        temp_filename = f.name

    try:
        data_dict = {'test_array': arr}
        matlab.savemat(temp_filename, data_dict)
        loaded_dict = matlab.loadmat(temp_filename)

        assert 'test_array' in loaded_dict
        assert loaded_dict['test_array'].shape == arr.shape, \
            f"Shape mismatch: original {arr.shape}, loaded {loaded_dict['test_array'].shape}"
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

if __name__ == "__main__":
    test_roundtrip_1d_arrays()