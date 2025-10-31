import numpy as np
import numpy.ctypeslib

# Create a simple 2x2 integer array
arr = np.array([[1, 2], [3, 4]], dtype=np.int32)

# Convert to Fortran-ordered (column-major) array
f_arr = np.asfortranarray(arr)

# Check array flags
print("Original array (C-contiguous):")
print(f"  Flags: C_CONTIGUOUS={arr.flags.c_contiguous}, F_CONTIGUOUS={arr.flags.f_contiguous}")
print(f"  __array_interface__['strides'] = {arr.__array_interface__['strides']}")

print("\nFortran-ordered array:")
print(f"  Flags: C_CONTIGUOUS={f_arr.flags.c_contiguous}, F_CONTIGUOUS={f_arr.flags.f_contiguous}")
print(f"  __array_interface__['strides'] = {f_arr.__array_interface__['strides']}")

# Try to convert C-contiguous array to ctypes (should work)
print("\nConverting C-contiguous array to ctypes:")
try:
    c_ctypes = np.ctypeslib.as_ctypes(arr)
    print(f"  Success! Type: {type(c_ctypes)}")
except TypeError as e:
    print(f"  Failed: {e}")

# Try to convert F-contiguous array to ctypes (will fail with current implementation)
print("\nConverting F-contiguous array to ctypes:")
try:
    f_ctypes = np.ctypeslib.as_ctypes(f_arr)
    print(f"  Success! Type: {type(f_ctypes)}")
except TypeError as e:
    print(f"  Failed: {e}")