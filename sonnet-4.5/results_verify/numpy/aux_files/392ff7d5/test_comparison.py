import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.replace vs numpy.strings.center dtype behavior:")
print("-" * 60)

# Test 1: Replace that would expand string
print("Test 1: Replace '0' with '00' in array ['0']")
arr1 = np.array(['0'], dtype='<U1')
print(f"Input: {arr1}, dtype: {arr1.dtype}")
result1 = nps.replace(arr1, '0', '00')
print(f"Result: {result1}, dtype: {result1.dtype}")
print(f"Expected: ['00'], Actual: {result1}")
print()

# Test 2: Center that expands string
print("Test 2: Center 'a' to width 5")
arr2 = np.array(['a'], dtype='<U1')
print(f"Input: {arr2}, dtype: {arr2.dtype}")
result2 = nps.center(arr2, 5, 'x')
print(f"Result: {result2}, dtype: {result2.dtype}")
print(f"Expected: ['xxaxx'], Actual: {result2}")
print()

# Test 3: Another replace test with longer strings
print("Test 3: Replace 'ab' with 'abcdef' in array ['ab']")
arr3 = np.array(['ab'], dtype='<U2')
print(f"Input: {arr3}, dtype: {arr3.dtype}")
result3 = nps.replace(arr3, 'ab', 'abcdef')
print(f"Result: {result3}, dtype: {result3.dtype}")
print(f"Expected: ['abcdef'], Actual: {result3}")
print()

# Test 4: Python string replacement for comparison
print("Test 4: Python native string replace")
py_str = '0'
py_result = py_str.replace('0', '00')
print(f"Input: '{py_str}'")
print(f"Result: '{py_result}'")
print()

# Test 5: Test with explicit larger dtype
print("Test 5: Replace with pre-allocated larger dtype")
arr5 = np.array(['0'], dtype='<U10')
print(f"Input: {arr5}, dtype: {arr5.dtype}")
result5 = nps.replace(arr5, '0', '00')
print(f"Result: {result5}, dtype: {result5.dtype}")
print(f"Expected: ['00'], Actual: {result5}")
print()

# Test 6: Test multiple replacements
print("Test 6: Multiple replacements in longer string")
arr6 = np.array(['hello'], dtype='<U5')
print(f"Input: {arr6}, dtype: {arr6.dtype}")
result6 = nps.replace(arr6, 'l', 'LL')
print(f"Result: {result6}, dtype: {result6.dtype}")
print(f"Expected: ['heLLLLo'] but truncated?, Actual: {result6}")
print(f"Python equivalent: {'hello'.replace('l', 'LL')}")