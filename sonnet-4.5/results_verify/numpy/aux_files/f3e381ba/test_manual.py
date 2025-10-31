import numpy as np
import numpy.ctypeslib

# Create a regular C-contiguous array
arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
print(f"Original array:\n{arr}")
print(f"C-contiguous: {arr.flags.c_contiguous}")
print(f"F-contiguous: {arr.flags.f_contiguous}")

# Convert to Fortran order
f_arr = np.asfortranarray(arr)
print(f"\nFortran-ordered array:\n{f_arr}")
print(f"C-contiguous: {f_arr.flags.c_contiguous}")
print(f"F-contiguous: {f_arr.flags.f_contiguous}")

# Try to convert C-contiguous array to ctypes
print("\nTrying as_ctypes on C-contiguous array...")
try:
    c_ct = np.ctypeslib.as_ctypes(arr)
    print("Success! Type:", type(c_ct))
except Exception as e:
    print(f"Failed: {e}")

# Try to convert F-contiguous array to ctypes
print("\nTrying as_ctypes on F-contiguous array...")
try:
    f_ct = np.ctypeslib.as_ctypes(f_arr)
    print("Success! Type:", type(f_ct))
except Exception as e:
    print(f"Failed: {e}")

# Let's check the __array_interface__ for both
print(f"\nC-array __array_interface__ strides: {arr.__array_interface__['strides']}")
print(f"F-array __array_interface__ strides: {f_arr.__array_interface__['strides']}")