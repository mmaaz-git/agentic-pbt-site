import numpy.rec
import numpy as np
import traceback

print("Testing numpy.rec.fromrecords with empty tuples:")
print("-" * 50)

# Test case 1: List of empty tuples
print("\n1. Testing with list of empty tuples [(), (), ()]")
try:
    records = [(), (), ()]
    rec_arr = numpy.rec.fromrecords(records)
    print(f"Success! Created array of shape: {rec_arr.shape}")
except Exception as e:
    print(f"Failed with: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test case 2: Single empty tuple
print("\n2. Testing with single empty tuple [()]")
try:
    records = [()]
    rec_arr = numpy.rec.fromrecords(records)
    print(f"Success! Created array of shape: {rec_arr.shape}")
except Exception as e:
    print(f"Failed with: {type(e).__name__}: {e}")

# Test case 3: NumPy supports empty structured types
print("\n3. Testing np.zeros(5, dtype=[]) for comparison")
try:
    arr = np.zeros(5, dtype=[])
    print(f"Success! Created empty structured array of shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
except Exception as e:
    print(f"Failed with: {type(e).__name__}: {e}")

# Test case 4: fromarrays with empty list
print("\n4. Testing numpy.rec.fromarrays([]) for comparison")
try:
    rec_arr = numpy.rec.fromarrays([])
    print(f"Success! Created array from empty arrays list of shape: {rec_arr.shape}")
except Exception as e:
    print(f"Failed with: {type(e).__name__}: {e}")