import ctypes
import numpy as np

print("Testing numpy.ctypeslib.as_array shape bug reproduction")
print("=" * 60)

# Test case 1: The specific case from the bug report
arr = np.array([[False]], dtype=np.bool_)
print(f"Original array shape: {arr.shape}")
print(f"Original array: {arr}")

ct_arr = np.ctypeslib.as_ctypes(arr)
print(f"ctypes array type: {type(ct_arr)}")
print(f"ctypes array _type_: {ct_arr._type_}")

ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))
print(f"Pointer type: {type(ptr)}")
print(f"Pointer _type_: {ptr._type_}")

result = np.ctypeslib.as_array(ptr, shape=(1, 1))
print(f"\nExpected shape: (1, 1)")
print(f"Actual shape: {result.shape}")
print(f"Result array: {result}")
print(f"Arrays equal? {np.array_equal(result, arr)}")

print("\n" + "=" * 60)
print("Additional test cases:")

# Test case 2: 2x2 array
arr2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
print(f"\nOriginal array shape: {arr2.shape}")
ct_arr2 = np.ctypeslib.as_ctypes(arr2)
ptr2 = ctypes.cast(ct_arr2, ctypes.POINTER(ct_arr2._type_))
result2 = np.ctypeslib.as_array(ptr2, shape=(2, 2))
print(f"Expected shape: (2, 2)")
print(f"Actual shape: {result2.shape}")

# Test case 3: 1D array (should work correctly)
arr3 = np.array([1, 2, 3, 4], dtype=np.int32)
print(f"\nOriginal 1D array shape: {arr3.shape}")
ct_arr3 = np.ctypeslib.as_ctypes(arr3)
ptr3 = ctypes.cast(ct_arr3, ctypes.POINTER(ct_arr3._type_))
result3 = np.ctypeslib.as_array(ptr3, shape=(4,))
print(f"Expected shape: (4,)")
print(f"Actual shape: {result3.shape}")
print(f"1D case works correctly? {result3.shape == (4,)}")

# Test case 4: 3D array
arr4 = np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)
print(f"\nOriginal 3D array shape: {arr4.shape}")
ct_arr4 = np.ctypeslib.as_ctypes(arr4)
ptr4 = ctypes.cast(ct_arr4, ctypes.POINTER(ct_arr4._type_))
result4 = np.ctypeslib.as_array(ptr4, shape=(2, 1, 2))
print(f"Expected shape: (2, 1, 2)")
print(f"Actual shape: {result4.shape}")