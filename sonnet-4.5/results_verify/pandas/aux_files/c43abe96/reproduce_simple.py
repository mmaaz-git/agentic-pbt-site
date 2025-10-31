import pandas.arrays as arrays
import numpy as np

np_arr = np.array([0])
numpy_ext_arr = arrays.NumpyExtensionArray(np_arr)

print(f"numpy_ext_arr.dtype: {numpy_ext_arr.dtype}")
print(f"np_arr.dtype: {np_arr.dtype}")
print(f"Type of numpy_ext_arr.dtype: {type(numpy_ext_arr.dtype)}")
print(f"Type of np_arr.dtype: {type(np_arr.dtype)}")
print(f"String repr are same: {str(numpy_ext_arr.dtype) == str(np_arr.dtype)}")
print(f"Are they equal? {numpy_ext_arr.dtype == np_arr.dtype}")

# Check if numpy_dtype attribute exists
if hasattr(numpy_ext_arr.dtype, 'numpy_dtype'):
    print(f"\nnumpy_ext_arr.dtype.numpy_dtype: {numpy_ext_arr.dtype.numpy_dtype}")
    print(f"numpy_ext_arr.dtype.numpy_dtype == np_arr.dtype: {numpy_ext_arr.dtype.numpy_dtype == np_arr.dtype}")

print("\nAssertion will fail:")
assert numpy_ext_arr.dtype == np_arr.dtype