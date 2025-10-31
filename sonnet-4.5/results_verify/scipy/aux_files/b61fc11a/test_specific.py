import scipy.io.matlab as sio_matlab
import numpy as np
import tempfile
import os

# Test the specific failing input from the bug report
data = {'A': np.array([0.+1j*np.inf])}

print(f"Original data: {data}")
print(f"Original A real: {data['A'][0].real}")
print(f"Original A imag: {data['A'][0].imag}")

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    sio_matlab.savemat(fname, data, format='4')
    loaded = sio_matlab.loadmat(fname)

    print(f"\nLoaded data: {loaded}")

    for key in data.keys():
        assert key in loaded, f"Key {key} not found in loaded data"
        original = data[key]
        result = loaded[key]

        if original.ndim == 0:
            expected_shape = (1, 1)
        elif original.ndim == 1:
            expected_shape = (1, original.shape[0])
        else:
            expected_shape = original.shape

        print(f"Original shape: {original.shape}, Result shape: {result.shape}, Expected: {expected_shape}")
        assert result.shape == expected_shape, f"Shape mismatch: {result.shape} != {expected_shape}"

        original_reshaped = original.reshape(expected_shape)

        print(f"Original reshaped: {original_reshaped}")
        print(f"Result: {result}")
        print(f"Result real: {result[0,0].real}")
        print(f"Result imag: {result[0,0].imag}")

        try:
            np.testing.assert_array_equal(result, original_reshaped)
            print("Arrays are equal")
        except AssertionError as e:
            print(f"Arrays NOT equal: {e}")
            print(f"Bug confirmed: 0.+infj corrupted to nan+infj")
finally:
    if os.path.exists(fname):
        os.remove(fname)