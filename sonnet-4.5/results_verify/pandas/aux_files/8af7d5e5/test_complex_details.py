import numpy as np

# Test how complex64 is currently handled
c64 = np.array([1+2j], dtype=np.complex64)
c128 = np.array([1+2j], dtype=np.complex128)

print("Complex64 array details:")
print(f"  dtype: {c64.dtype}")
print(f"  itemsize: {c64.dtype.itemsize}")
print(f"  value: {c64[0]}")

print("\nComplex128 array details:")
print(f"  dtype: {c128.dtype}")
print(f"  itemsize: {c128.dtype.itemsize}")
print(f"  value: {c128[0]}")

# See how complex64 is viewed as uint64
c64_as_uint = c64.view('u8')
print(f"\nComplex64 viewed as uint64: {c64_as_uint[0]}")

# Check if np.issubdtype works as expected
print(f"\nnp.issubdtype(np.complex64, np.complex128): {np.issubdtype(np.complex64, np.complex128)}")
print(f"np.issubdtype(np.complex64, np.complexfloating): {np.issubdtype(np.complex64, np.complexfloating)}")
print(f"np.issubdtype(np.complex128, np.complexfloating): {np.issubdtype(np.complex128, np.complexfloating)}")

# Check if complex64 falls into the generic numeric path
print(f"\nissubclass(np.complex64, np.number): {issubclass(np.complex64, np.number)}")
print(f"np.dtype('complex64').itemsize <= 8: {np.dtype('complex64').itemsize <= 8}")