from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays, array_shapes
import scipy.io.matlab as sio_matlab
import numpy as np
import tempfile
import os

@given(
    data=st.dictionaries(
        keys=st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,30}', fullmatch=True),
        values=arrays(
            dtype=st.sampled_from([np.float64, np.int32, np.uint8, np.complex128]),
            shape=array_shapes(max_dims=2, max_side=20),
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=100)
def test_savemat_loadmat_roundtrip_format4(data):
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        sio_matlab.savemat(fname, data, format='4')
        loaded = sio_matlab.loadmat(fname)

        for key in data.keys():
            assert key in loaded
            original = data[key]
            result = loaded[key]

            if original.ndim == 0:
                expected_shape = (1, 1)
            elif original.ndim == 1:
                expected_shape = (1, original.shape[0])
            else:
                expected_shape = original.shape

            assert result.shape == expected_shape

            original_reshaped = original.reshape(expected_shape)
            np.testing.assert_array_equal(result, original_reshaped)
    finally:
        if os.path.exists(fname):
            os.remove(fname)

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    test_savemat_loadmat_roundtrip_format4()
    print("All tests passed!")