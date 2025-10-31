#!/usr/bin/env python3

import numpy as np
import numpy.rec

# Test edge cases
test_cases = [
    ("Empty list", []),
    ("List with one integer", [1]),
    ("List with multiple integers", [1, 2, 3]),
    ("List with one tuple", [(1, 2)]),
    ("List with multiple tuples", [(1, 2), (3, 4)]),
    ("Empty tuple", ()),
    ("Tuple with one integer", (1,)),
    ("Tuple with multiple integers", (1, 2, 3)),
]

for name, obj in test_cases:
    print(f"\nTesting {name}: {obj}")
    try:
        result = numpy.rec.array(obj)
        print(f"  Success! Type: {type(result)}, Shape: {result.shape}, Dtype: {result.dtype}")
    except Exception as e:
        print(f"  Failed with {type(e).__name__}: {e}")

# Test what regular numpy.array does with these
print("\n" + "="*50)
print("Testing regular numpy.array with empty list:")
result = np.array([])
print(f"  Type: {type(result)}, Shape: {result.shape}, Dtype: {result.dtype}")

# Test what the dispatched functions do directly
print("\n" + "="*50)
print("Direct testing of dispatched functions:")

print("\nnumpy.rec.fromrecords([]):")
try:
    result = numpy.rec.fromrecords([])
    print(f"  Success! Type: {type(result)}, Shape: {result.shape}, Dtype: {result.dtype}")
except Exception as e:
    print(f"  Failed with {type(e).__name__}: {e}")

print("\nnumpy.rec.fromarrays([]):")
try:
    result = numpy.rec.fromarrays([])
    print(f"  Success! Type: {type(result)}, Shape: {result.shape}, Dtype: {result.dtype}")
except Exception as e:
    print(f"  Failed with {type(e).__name__}: {e}")

# What about with empty tuples?
print("\nnumpy.rec.fromrecords(()):")
try:
    result = numpy.rec.fromrecords(())
    print(f"  Success! Type: {type(result)}, Shape: {result.shape}, Dtype: {result.dtype}")
except Exception as e:
    print(f"  Failed with {type(e).__name__}: {e}")

print("\nnumpy.rec.fromarrays(()):")
try:
    result = numpy.rec.fromarrays(())
    print(f"  Success! Type: {type(result)}, Shape: {result.shape}, Dtype: {result.dtype}")
except Exception as e:
    print(f"  Failed with {type(e).__name__}: {e}")