import numpy as np
import numpy.ctypeslib

print("Testing negative ndim with ndpointer...")
ptr = numpy.ctypeslib.ndpointer(ndim=-1)
print(f"Created: {ptr}")
print(f"ndim: {ptr._ndim_}")

arr = np.array([1, 2, 3])
print(f"Testing with array: {arr}")
try:
    ptr.from_param(arr)
except TypeError as e:
    print(f"Error: {e}")

# Test with other negative values
print("\nTesting with ndim=-10:")
ptr2 = numpy.ctypeslib.ndpointer(ndim=-10)
print(f"Created: {ptr2}")
print(f"ndim: {ptr2._ndim_}")

try:
    ptr2.from_param(arr)
except TypeError as e:
    print(f"Error: {e}")