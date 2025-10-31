import numpy as np
import numpy.ctypeslib as npc

# Create a structured array
arr = np.array([(1, 2.0)], dtype=[('x', np.int32), ('y', np.float64)])
print(f"Original array: {arr}")
print(f"Array dtype: {arr.dtype}")
print(f"Array flags: C_CONTIGUOUS={arr.flags.c_contiguous}, WRITEABLE={arr.flags.writeable}")
print()

# Show that as_ctypes_type works on the structured dtype
try:
    ctype = npc.as_ctypes_type(arr.dtype)
    print(f"as_ctypes_type(arr.dtype) works: {ctype}")
    print(f"Created ctypes structure fields: {[field[0] for field in ctype._fields_]}")
    print()
except Exception as e:
    print(f"as_ctypes_type failed: {e}")
    print()

# Now try as_ctypes on the structured array - this will fail
try:
    print("Attempting npc.as_ctypes(arr)...")
    c_arr = npc.as_ctypes(arr)
    print(f"Success: {c_arr}")
except NotImplementedError as e:
    print(f"NotImplementedError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")