import numpy as np
import numpy.fft._pocketfft_umath as pfu

# Test what types the rfft ufuncs accept
print("Testing rfft ufunc types...")
print()
print("rfft_n_even types:", pfu.rfft_n_even.types)
print("rfft_n_odd types:", pfu.rfft_n_odd.types)
print()

# Test with different dtypes
test_arrays = [
    ("float32", np.array([1.0, 2.0], dtype=np.float32)),
    ("float64", np.array([1.0, 2.0], dtype=np.float64)),
    ("complex64", np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex64)),
    ("complex128", np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex128)),
]

for dtype_name, arr in test_arrays:
    print(f"Testing with {dtype_name}:")
    print(f"  Array: {arr}")
    print(f"  Dtype: {arr.dtype}")
    try:
        result = np.fft.rfft(arr)
        print(f"  Result: {result}")
        print(f"  Result dtype: {result.dtype}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    print()