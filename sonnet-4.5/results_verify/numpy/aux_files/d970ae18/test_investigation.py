import ctypes
import numpy as np

print("Deep investigation of the as_array behavior")
print("=" * 60)

# Test with a simple 2D array
arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
print(f"Original array:\n{arr}")
print(f"Original shape: {arr.shape}")

# Convert to ctypes
ct_arr = np.ctypeslib.as_ctypes(arr)
print(f"\nctypes array type: {type(ct_arr)}")
print(f"ctypes array type name: {type(ct_arr).__name__}")

# The ct_arr._type_ is the type of the first dimension
print(f"ct_arr._type_: {ct_arr._type_}")
print(f"ct_arr._type_ is already an array type: {hasattr(ct_arr._type_, '_type_')}")

if hasattr(ct_arr._type_, '_type_'):
    print(f"ct_arr._type_._type_ (element type): {ct_arr._type_._type_}")

# Create pointer to this array
ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))
print(f"\nPointer type: {type(ptr)}")
print(f"Pointer _type_: {ptr._type_}")

# When we call as_array with this pointer and shape (2, 2)
# The implementation does _ctype_ndarray(ptr._type_, (2, 2))
# But ptr._type_ is already c_int_Array_2, not c_int
# So it creates c_int_Array_2_Array_2_Array_2 instead of c_int_Array_2_Array_2

print("\nSimulating what _ctype_ndarray does:")
print(f"Input element_type: {ptr._type_}")
print(f"Input shape: (2, 2)")

# This is what _ctype_ndarray effectively does:
element_type = ptr._type_  # This is c_int_Array_2
for dim in (2, 2)[::-1]:  # Iterate [2, 2] in reverse
    element_type = dim * element_type
    print(f"After applying dim {dim}: {element_type}")

print("\nWhat SHOULD happen:")
print("If ptr._type_ is already an array, we should extract the base scalar type first")

# Find the base scalar type
base_type = ct_arr._type_
while hasattr(base_type, '_type_'):
    base_type = base_type._type_
print(f"Base scalar type: {base_type}")

print("\n" + "=" * 60)
print("Testing the documented examples:")

# Example 1: Simple scalar pointer (from docs)
buffer = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_int))
print(f"\nExample 1 - Scalar pointer to c_int:")
print(f"pointer._type_: {pointer._type_}")
print(f"Has _type_ attribute? {hasattr(pointer._type_, '_type_')}")
np_array = np.ctypeslib.as_array(pointer, (5,))
print(f"Result shape: {np_array.shape} (correct)")

# Now contrast with array type pointer
buffer2d = (ctypes.c_int * 2 * 2)((1, 2), (3, 4))
pointer2d = ctypes.cast(buffer2d, ctypes.POINTER(buffer2d._type_))
print(f"\nExample 2 - Pointer to array type:")
print(f"pointer2d._type_: {pointer2d._type_}")
print(f"Has _type_ attribute? {hasattr(pointer2d._type_, '_type_')}")
result = np.ctypeslib.as_array(pointer2d, (2, 2))
print(f"Result shape: {result.shape} (incorrect - extra dimension added)")