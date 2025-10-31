import numpy as np
import numpy.ctypeslib as npc

arr = np.array([(1, 2.0)], dtype=[('x', np.int32), ('y', np.float64)])

ctype = npc.as_ctypes_type(arr.dtype)
print(f"as_ctypes_type works: {ctype}")

try:
    c_arr = npc.as_ctypes(arr)
    print(f"as_ctypes succeeded: {c_arr}")
except Exception as e:
    print(f"as_ctypes failed with: {type(e).__name__}: {e}")