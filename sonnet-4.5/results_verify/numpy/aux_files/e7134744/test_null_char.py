import numpy as np
import numpy.strings as nps

# Test with null character
print("Testing null character replacement:")
print("="*50)

# Test 1: Empty string
arr = np.array([''], dtype=str)
result = nps.replace(arr, '\x00', 'X')
expected = ''.replace('\x00', 'X')
print(f"Test 1: Empty string")
print(f"  Input: {repr(arr[0])}")
print(f"  Replace '\\x00' with 'X'")
print(f"  Expected: {repr(expected)}")
print(f"  Got: {repr(result[0])}")
print(f"  Match: {result[0] == expected}")
print()

# Test 2: Non-empty string without null chars
arr = np.array(['abc'], dtype=str)
result = nps.replace(arr, '\x00', 'X')
expected = 'abc'.replace('\x00', 'X')
print(f"Test 2: String 'abc' without null chars")
print(f"  Input: {repr(arr[0])}")
print(f"  Replace '\\x00' with 'X'")
print(f"  Expected: {repr(expected)}")
print(f"  Got: {repr(result[0])}")
print(f"  Match: {result[0] == expected}")
print()

# Test 3: String with actual null character
arr = np.array(['a\x00b'], dtype=str)
result = nps.replace(arr, '\x00', 'X')
expected = 'a\x00b'.replace('\x00', 'X')
print(f"Test 3: String with actual null character")
print(f"  Input: {repr(arr[0])}")
print(f"  Replace '\\x00' with 'X'")
print(f"  Expected: {repr(expected)}")
print(f"  Got: {repr(result[0])}")
print(f"  Match: {result[0] == expected}")
print()

# Test 4: Multiple null characters
arr = np.array(['\x00\x00'], dtype=str)
result = nps.replace(arr, '\x00', 'X')
expected = '\x00\x00'.replace('\x00', 'X')
print(f"Test 4: String with two null characters")
print(f"  Input: {repr(arr[0])}")
print(f"  Replace '\\x00' with 'X'")
print(f"  Expected: {repr(expected)}")
print(f"  Got: {repr(result[0])}")
print(f"  Match: {result[0] == expected}")
print()

# Test 5: With count parameter
arr = np.array(['abc'], dtype=str)
result = nps.replace(arr, '\x00', 'X', count=1)
expected = 'abc'.replace('\x00', 'X', 1)
print(f"Test 5: String 'abc' with count=1")
print(f"  Input: {repr(arr[0])}")
print(f"  Replace '\\x00' with 'X', count=1")
print(f"  Expected: {repr(expected)}")
print(f"  Got: {repr(result[0])}")
print(f"  Match: {result[0] == expected}")