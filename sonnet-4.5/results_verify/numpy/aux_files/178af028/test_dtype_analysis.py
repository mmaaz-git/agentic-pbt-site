import numpy as np
import numpy.strings as nps

print("Analyzing dtype behavior in numpy.strings.replace")
print("="*60)

# Test case 1: The failing case
print("\nTest 1: Input dtype U1, replacement longer than input")
print("-"*40)
arr = np.array(['0'])
print(f"Input array: {arr}")
print(f"Input dtype: {arr.dtype}")
print(f"Input max string length: {arr.dtype.str}")

result = nps.replace(arr, '0', '00', count=1)
print(f"Result: {result}")
print(f"Result dtype: {result.dtype}")
print(f"Result max string length: {result.dtype.str}")
print(f"Expected result: ['00']")
print(f"Actual result matches expected: {str(result[0]) == '00'}")

# Test case 2: Input with sufficient dtype size
print("\nTest 2: Input dtype U2, replacement fits")
print("-"*40)
arr2 = np.array(['0'], dtype='U2')
print(f"Input array: {arr2}")
print(f"Input dtype: {arr2.dtype}")

result2 = nps.replace(arr2, '0', '00', count=1)
print(f"Result: {result2}")
print(f"Result dtype: {result2.dtype}")
print(f"Result matches expected '00': {str(result2[0]) == '00'}")

# Test case 3: Check what happens with the intermediate values
print("\nTest 3: Debugging intermediate values")
print("-"*40)
arr3 = np.array(['0'])
old_str = '0'
new_str = '00'

print(f"Original old string: '{old_str}', len={len(old_str)}")
print(f"Original new string: '{new_str}', len={len(new_str)}")

# Try to mimic what the code does
old_arr = np.asarray(old_str)
new_arr = np.asarray(new_str)
print(f"old_arr dtype: {old_arr.dtype}")
print(f"new_arr dtype: {new_arr.dtype}")

# When cast to U1
old_u1 = old_arr.astype('U1')
new_u1 = new_arr.astype('U1')
print(f"old cast to U1: '{old_u1}', len={len(str(old_u1))}")
print(f"new cast to U1: '{new_u1}', len={len(str(new_u1))}")

# Test case 4: Multiple replacements with growing string
print("\nTest 4: Multiple elements with dtype U1")
print("-"*40)
arr4 = np.array(['1', '2', '3'])
print(f"Input array: {arr4}")
print(f"Input dtype: {arr4.dtype}")

result4 = nps.replace(arr4, '1', '111', count=1)
print(f"Result: {result4}")
print(f"Result dtype: {result4.dtype}")
for i in range(len(arr4)):
    expected = arr4[i].replace('1', '111', 1)
    actual = str(result4[i])
    print(f"  arr[{i}]='{arr4[i]}' -> expected='{expected}', actual='{actual}', match={actual == expected}")