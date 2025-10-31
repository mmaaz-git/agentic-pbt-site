import numpy as np
import numpy.strings as nps

print("=== Testing numpy.strings.replace behavior ===\n")

# Test 1: Basic replacement that increases length
print("Test 1: Basic replacement increasing length")
s = '0'
arr = np.array([s])
result = nps.replace(arr, '0', '00')
print(f"Input: '{s}', dtype: {arr.dtype}")
print(f"Replace '0' with '00'")
print(f"Result: '{result[0]}', dtype: {result.dtype}")
print(f"Python str.replace: '{s.replace('0', '00')}'")
print()

# Test 2: Multiple replacements
print("Test 2: Multiple replacements")
s = 'abc'
arr = np.array([s])
result = nps.replace(arr, 'a', 'XXX')
print(f"Input: '{s}', dtype: {arr.dtype}")
print(f"Replace 'a' with 'XXX'")
print(f"Result: '{result[0]}', dtype: {result.dtype}")
print(f"Python str.replace: '{s.replace('a', 'XXX')}'")
print()

# Test 3: Replacement that would exceed original length significantly
print("Test 3: Large expansion")
s = 'x'
arr = np.array([s])
result = nps.replace(arr, 'x', 'abcdefghij')
print(f"Input: '{s}', dtype: {arr.dtype}")
print(f"Replace 'x' with 'abcdefghij'")
print(f"Result: '{result[0]}', dtype: {result.dtype}")
print(f"Python str.replace: '{s.replace('x', 'abcdefghij')}'")
print()

# Test 4: Array with explicit dtype
print("Test 4: Explicit dtype")
arr = np.array(['a'], dtype='<U10')
result = nps.replace(arr, 'a', 'aaaaa')
print(f"Input dtype: {arr.dtype}")
print(f"Result: '{result[0]}', dtype: {result.dtype}")
print(f"Expected: 'aaaaa'")
print()

# Test 5: Multiple elements in array
print("Test 5: Multiple elements")
arr = np.array(['a', 'b', 'c'])
result = nps.replace(arr, 'a', 'XXX')
print(f"Input: {arr}, dtype: {arr.dtype}")
print(f"Result: {result}, dtype: {result.dtype}")
print()

# Test 6: Compare with other string operations
print("Test 6: Compare with ljust (which handles dtype correctly)")
arr = np.array(['a'])
result_ljust = nps.ljust(arr, 5, fillchar='a')
result_replace = nps.replace(arr, 'a', 'aaaaa')
print(f"ljust to 5 chars: '{result_ljust[0]}', dtype: {result_ljust.dtype}")
print(f"replace 'a' with 'aaaaa': '{result_replace[0]}', dtype: {result_replace.dtype}")
print()

# Test 7: Check if StringDType is available (numpy 2.0+)
print("Test 7: Check StringDType availability")
try:
    from numpy import StringDType
    print("StringDType is available")
    arr_str = np.array(['a'], dtype=StringDType())
    result_str = nps.replace(arr_str, 'a', 'aaaaa')
    print(f"With StringDType: '{result_str[0]}', dtype: {result_str.dtype}")
except ImportError:
    print("StringDType not available in this numpy version")