import numpy as np

# Check how numpy handles out-of-bounds indexing in regular arrays
arr = np.array([1, 2, 3])

print("=== Regular array indexing errors ===")
try:
    result = arr[5]
except IndexError as e:
    print(f"arr[5] raises: {e}")

# Check structured array field access
dt = np.dtype([('x', int), ('y', int)])
struct_arr = np.array([(1, 2), (3, 4)], dtype=dt)

print("\n=== Structured array field access ===")
try:
    result = struct_arr['z']  # non-existent field name
except Exception as e:
    print(f"struct_arr['z'] raises {type(e).__name__}: {e}")

# Check recarray attribute access
import numpy.rec
rec_arr = numpy.rec.fromarrays([[1, 2], [3, 4]], names='x,y')

print("\n=== Recarray attribute access ===")
try:
    result = rec_arr.z  # non-existent field attribute
except Exception as e:
    print(f"rec_arr.z raises {type(e).__name__}: {e}")