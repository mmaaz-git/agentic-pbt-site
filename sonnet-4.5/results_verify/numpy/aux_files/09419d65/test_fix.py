import numpy as np
import numpy.ctypeslib as npc

# Test the issue and simulate the fix

arr = np.array([(1, 2.0), (3, 4.0)], dtype=[('x', np.int32), ('y', np.float64)])
print(f"Original array: {arr}")
print(f"Array dtype: {arr.dtype}")

# Get the __array_interface__
ai = arr.__array_interface__
print(f"\n__array_interface__ typestr: {ai['typestr']}")
print(f"This is a void type, which causes the NotImplementedError")

# Show that using the actual dtype works
print(f"\nUsing arr.dtype directly:")
ctype_from_dtype = npc.as_ctypes_type(arr.dtype)
print(f"as_ctypes_type(arr.dtype) works: {ctype_from_dtype}")

# Show that using typestr fails
print(f"\nUsing typestr from __array_interface__:")
try:
    ctype_from_typestr = npc.as_ctypes_type(ai["typestr"])
    print(f"as_ctypes_type(typestr) works: {ctype_from_typestr}")
except Exception as e:
    print(f"as_ctypes_type(typestr) fails: {type(e).__name__}: {e}")

# Verify that the proposed fix logic makes sense
print(f"\nProposed fix logic:")
if hasattr(arr, 'dtype'):
    print(f"Object has dtype attribute, use it: {arr.dtype}")
    ctype_scalar = npc.as_ctypes_type(arr.dtype)
    print(f"Result: {ctype_scalar}")
else:
    print(f"Object doesn't have dtype, fall back to typestr")

# Test that regular arrays still work
regular_arr = np.array([1, 2, 3], dtype=np.int32)
print(f"\n\nRegular array test:")
print(f"Array: {regular_arr}")
print(f"dtype: {regular_arr.dtype}")
try:
    c_arr = npc.as_ctypes(regular_arr)
    print(f"as_ctypes works: {c_arr}")
    print(f"Type: {type(c_arr)}")
except Exception as e:
    print(f"Failed: {e}")