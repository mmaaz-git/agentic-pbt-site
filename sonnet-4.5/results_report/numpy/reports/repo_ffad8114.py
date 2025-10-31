import numpy as np
import numpy.ctypeslib

# Test with negative ndim value
ptr = numpy.ctypeslib.ndpointer(ndim=-1)
print(f"Created pointer class: {ptr}")
print(f"Class name: {ptr.__name__}")
print(f"ndim attribute: {ptr._ndim_}")

# Try to use the pointer with an actual array
arr = np.array([1, 2, 3])
print(f"\nAttempting to validate array with shape {arr.shape} and ndim {arr.ndim}")
try:
    result = ptr.from_param(arr)
    print(f"Validation succeeded: {result}")
except TypeError as e:
    print(f"Validation failed with error: {e}")

# Test with another negative value
print("\n--- Testing with ndim=-10 ---")
ptr2 = numpy.ctypeslib.ndpointer(ndim=-10)
print(f"Created pointer class: {ptr2}")
print(f"Class name: {ptr2.__name__}")
print(f"ndim attribute: {ptr2._ndim_}")

try:
    result = ptr2.from_param(arr)
    print(f"Validation succeeded: {result}")
except TypeError as e:
    print(f"Validation failed with error: {e}")