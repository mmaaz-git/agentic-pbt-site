import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Test 1: Create array with None and integer
print("Test 1: Array with None and integer")
arr1 = ArrowExtensionArray(pa.array([None, 42]))
print(f"Type: {arr1._pa_array.type}")
print(f"Array: {arr1}")
print()

# Test 2: Create array with only None
print("Test 2: Array with only None")
arr2 = ArrowExtensionArray(pa.array([None]))
print(f"Type: {arr2._pa_array.type}")
print(f"Array: {arr2}")
print()

# Test 3: Try to cast null array to int64
print("Test 3: Attempt to cast null array to int64")
null_arr = pa.array([None])
print(f"Null array type: {null_arr.type}")
try:
    int_arr = null_arr.cast(pa.int64())
    print(f"Cast successful: {int_arr}")
except Exception as e:
    print(f"Cast failed: {e}")
print()

# Test 4: Create int64 array with None values
print("Test 4: Create int64 array directly with None values")
int_arr_with_none = pa.array([None] * 3, type=pa.int64())
print(f"Type: {int_arr_with_none.type}")
print(f"Array: {int_arr_with_none}")
print()

# Test 5: Check if we can cast an integer to null type
print("Test 5: Try to cast integer to null type")
int_val = pa.array([42])
try:
    null_cast = int_val.cast(pa.null())
    print(f"Cast successful: {null_cast}")
except Exception as e:
    print(f"Cast failed: {e}")