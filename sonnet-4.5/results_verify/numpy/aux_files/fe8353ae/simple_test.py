import numpy as np
import numpy.strings as ns

# Test the specific failing case
arr = np.array(['a'])
result = ns.replace(arr, 'a', 'aa')

print("Input:", arr)
print("Result:", result)
print("Expected:", ['aa'])
print(f"Test passed: {result[0] == 'aa'}")

# Also test with Python's native replace to confirm expected behavior
native_result = 'a'.replace('a', 'aa')
print(f"\nPython native 'a'.replace('a', 'aa') = '{native_result}'")

# Test other similar cases
print("\nAdditional tests:")

# Test with string that has more than just the replacement char
arr2 = np.array(['ab'])
result2 = ns.replace(arr2, 'a', 'aa')
print(f"'ab' replace 'a' -> 'aa': {result2[0]} (expected: 'aab')")

# Test with multiple occurrences
arr3 = np.array(['aaa'])
result3 = ns.replace(arr3, 'a', 'aa')
print(f"'aaa' replace 'a' -> 'aa': {result3[0]} (expected: 'aaaaaa')")

# Test with single char to single char
arr4 = np.array(['a'])
result4 = ns.replace(arr4, 'a', 'b')
print(f"'a' replace 'a' -> 'b': {result4[0]} (expected: 'b')")

# Test replacing with shorter string
arr5 = np.array(['aa'])
result5 = ns.replace(arr5, 'aa', 'a')
print(f"'aa' replace 'aa' -> 'a': {result5[0]} (expected: 'a')")