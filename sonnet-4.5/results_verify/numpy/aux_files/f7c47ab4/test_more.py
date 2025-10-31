import numpy as np
import numpy.rec

# Test with single empty array vs empty list
print("Test 1: Single empty array in list:")
try:
    empty_arr = np.array([])
    result = numpy.rec.fromarrays([empty_arr], names='a')
    print(f"Success with single empty array: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"Failed: {e}")

print("\nTest 2: Multiple empty arrays:")
try:
    result = numpy.rec.fromarrays([np.array([]), np.array([])], names='a,b')
    print(f"Success with multiple empty arrays: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"Failed: {e}")

print("\nTest 3: Test with explicit dtype and shape for empty list:")
try:
    dtype = np.dtype([('a', 'i4')])
    result = numpy.rec.fromarrays([], dtype=dtype, shape=(0,))
    print(f"Success with dtype and shape: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"Failed: {e}")

print("\nTest 4: Check similar functions:")
# np.rec.fromrecords behavior
try:
    result = np.rec.fromrecords([], names='a')
    print(f"np.rec.fromrecords([]) works: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"np.rec.fromrecords([]) failed: {e}")

# np.rec.array behavior
try:
    result = np.rec.array([], names='a')
    print(f"np.rec.array([]) works: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"np.rec.array([]) failed: {e}")