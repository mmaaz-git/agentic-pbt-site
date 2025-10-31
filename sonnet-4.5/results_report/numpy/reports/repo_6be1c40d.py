import ctypes
import numpy as np

# Test case 1: 2D array with shape (1, 1)
print("Test case 1: 2D array with shape (1, 1)")
arr = np.array([[False]], dtype=np.bool_)
print(f"Original array: {arr}")
print(f"Original shape: {arr.shape}")

ct_arr = np.ctypeslib.as_ctypes(arr)
print(f"ctypes array type: {type(ct_arr)}")

ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))
print(f"Pointer type: {type(ptr)}")
print(f"Pointer element type: {ptr._type_}")

result = np.ctypeslib.as_array(ptr, shape=(1, 1))
print(f"Result array: {result}")
print(f"Result shape: {result.shape}")
print(f"Expected shape: (1, 1)")
print(f"Shape mismatch: {result.shape != (1, 1)}")
print()

# Test case 2: 2D array with shape (2, 2)
print("Test case 2: 2D array with shape (2, 2)")
arr2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
print(f"Original array:\n{arr2}")
print(f"Original shape: {arr2.shape}")

ct_arr2 = np.ctypeslib.as_ctypes(arr2)
ptr2 = ctypes.cast(ct_arr2, ctypes.POINTER(ct_arr2._type_))
result2 = np.ctypeslib.as_array(ptr2, shape=(2, 2))
print(f"Result array:\n{result2}")
print(f"Result shape: {result2.shape}")
print(f"Expected shape: (2, 2)")
print(f"Shape mismatch: {result2.shape != (2, 2)}")
print()

# Test case 3: 1D array (should work correctly)
print("Test case 3: 1D array with shape (3,)")
arr3 = np.array([1, 2, 3], dtype=np.int32)
print(f"Original array: {arr3}")
print(f"Original shape: {arr3.shape}")

ct_arr3 = np.ctypeslib.as_ctypes(arr3)
ptr3 = ctypes.cast(ct_arr3, ctypes.POINTER(ct_arr3._type_))
result3 = np.ctypeslib.as_array(ptr3, shape=(3,))
print(f"Result array: {result3}")
print(f"Result shape: {result3.shape}")
print(f"Expected shape: (3,)")
print(f"Shape mismatch: {result3.shape != (3,)}")