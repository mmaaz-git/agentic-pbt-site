import numpy as np
import numpy.ctypeslib as npc
import ctypes

# Test that as_ctypes_type works with structured dtypes
dtype = np.dtype([('x', np.int32), ('y', np.float64)])
print(f"Testing dtype: {dtype}")

try:
    ctype = npc.as_ctypes_type(dtype)
    print(f"as_ctypes_type succeeded: {ctype}")

    # Check if it's actually a ctypes Structure
    print(f"Is Structure subclass: {issubclass(ctype, ctypes.Structure)}")

    # Check fields
    print(f"Fields: {ctype._fields_}")

    # Create an instance
    instance = ctype(1, 2.0)
    print(f"Created instance: x={instance.x}, y={instance.y}")

except Exception as e:
    print(f"as_ctypes_type failed: {type(e).__name__}: {e}")

# Also test with the array's dtype directly
arr = np.array([(1, 2.0), (3, 4.0)], dtype=[('x', np.int32), ('y', np.float64)])
print(f"\nArray dtype: {arr.dtype}")
try:
    ctype2 = npc.as_ctypes_type(arr.dtype)
    print(f"as_ctypes_type on array dtype succeeded: {ctype2}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")