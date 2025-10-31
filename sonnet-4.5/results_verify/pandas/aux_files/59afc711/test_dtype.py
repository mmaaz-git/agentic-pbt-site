import numpy.char as char
import numpy as np

# Test what happens with dtype
test_string = 'a'
result = char.replace(test_string, 'a', 'aa')

print(f"Input: {repr(test_string)}")
print(f"Output: {repr(result)}")
print(f"Output type: {type(result)}")
print(f"Output dtype: {result.dtype}")
print(f"Output shape: {result.shape}")
print(f"Item: {repr(result.item())}")

# Test with explicit array
arr = np.array(['a', 'b', 'c'])
print(f"\nArray dtype: {arr.dtype}")
result_arr = char.replace(arr, 'a', 'aa')
print(f"Result array dtype: {result_arr.dtype}")
print(f"Result array: {result_arr}")

# Test with longer strings
arr2 = np.array(['hello', 'world'])
print(f"\nArray2 dtype: {arr2.dtype}")
result_arr2 = char.replace(arr2, 'hello', 'hello world')
print(f"Result array2 dtype: {result_arr2.dtype}")
print(f"Result array2: {result_arr2}")