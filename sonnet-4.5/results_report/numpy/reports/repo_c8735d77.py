import numpy as np
import numpy.ctypeslib

# Test case 1: Create ndpointer with negative shape
ptr = numpy.ctypeslib.ndpointer(shape=(-1, 3))
print(f"Created pointer type: {ptr}")
print(f"Pointer shape attribute: {ptr._shape_}")

# Try to use this pointer with valid NumPy arrays
print("\nAttempting to use pointer with valid arrays:")

# Test with (2, 3) array
arr1 = np.zeros((2, 3))
print(f"Array shape (2, 3): {arr1.shape}")
try:
    ptr.from_param(arr1)
    print("  Accepted")
except TypeError as e:
    print(f"  Error: {e}")

# Test with (1, 3) array
arr2 = np.zeros((1, 3))
print(f"Array shape (1, 3): {arr2.shape}")
try:
    ptr.from_param(arr2)
    print("  Accepted")
except TypeError as e:
    print(f"  Error: {e}")

# Test case 2: Create ndpointer with multiple negative dimensions
ptr2 = numpy.ctypeslib.ndpointer(shape=(0, -1))
print(f"\nCreated second pointer with shape=(0, -1): {ptr2}")
print(f"Pointer shape attribute: {ptr2._shape_}")

# Test case 3: Show that numpy itself rejects negative dimensions
print("\nFor comparison, numpy array creation with negative shape:")
try:
    arr_negative = np.zeros((-1, 3))
    print(f"Created array: {arr_negative}")
except ValueError as e:
    print(f"NumPy error: {e}")