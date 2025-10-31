import numpy as np
import numpy.ctypeslib

# Test case 1: Create ndpointer with negative shape
ptr = numpy.ctypeslib.ndpointer(shape=(-1, 3))
print(f"Created: {ptr}")
print(f"shape: {ptr._shape_}")

# Try to use it with a valid array
arr = np.zeros((2, 3))
print(f"\nTrying to use with array of shape {arr.shape}")
try:
    result = ptr.from_param(arr)
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")

# Test case 2: Try with different array shapes
print("\nTrying different array shapes:")
test_arrays = [
    np.zeros((1, 3)),
    np.zeros((3, 3)),
    np.zeros((-1, 3)) if False else None,  # Can't create array with negative shape
]

for test_arr in test_arrays:
    if test_arr is not None:
        try:
            result = ptr.from_param(test_arr)
            print(f"  Shape {test_arr.shape}: Success")
        except TypeError as e:
            print(f"  Shape {test_arr.shape}: Error - {e}")

# Test case 3: Zero in shape
print("\n\nTest with zero in shape:")
ptr2 = numpy.ctypeslib.ndpointer(shape=(0, 5))
print(f"Created: {ptr2}")
print(f"shape: {ptr2._shape_}")

arr2 = np.zeros((0, 5))
print(f"Testing with array of shape {arr2.shape}")
try:
    result = ptr2.from_param(arr2)
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")