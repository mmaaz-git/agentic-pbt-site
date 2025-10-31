import numpy as np
import numpy.strings as nps

# Test case 1: The basic bug - slice(arr, 0, None) should return full string
print("Test 1: Basic bug demonstration")
print("=====================================")
arr = np.array(['hello'], dtype='U')
result = nps.slice(arr, 0, None)
print(f"nps.slice(np.array(['hello'], dtype='U'), 0, None)")
print(f"Result: {result}")
print(f"Expected: ['hello']")
print()

# Compare with Python's regular slicing
s = 'hello'
print(f"Python regular slicing: 'hello'[0:None] = '{s[0:None]}'")
print()

# Test case 2: Show the wrong behavior pattern
print("Test 2: Multiple examples showing the pattern")
print("=====================================")
test_strings = np.array(['abcdef', 'test', '12345'], dtype='U')
print(f"Input array: {test_strings}")
print()

# Case A: slice(arr, 2) - should be [:2]
result_a = nps.slice(test_strings, 2)
print(f"nps.slice(arr, 2) = {result_a}")
print(f"Expected (first 2 chars): ['ab', 'te', '12']")
print()

# Case B: slice(arr, 2, None) - should be [2:]
result_b = nps.slice(test_strings, 2, None)
print(f"nps.slice(arr, 2, None) = {result_b}")
print(f"Expected (from index 2 onwards): ['cdef', 'st', '345']")
print()

# Test case 3: The specific failing case from the bug report
print("Test 3: Specific failing case from bug report")
print("=====================================")
arr = np.array(['0'], dtype='<U1')
result = nps.slice(arr, 0, None)
print(f"nps.slice(np.array(['0'], dtype='<U1'), 0, None)")
print(f"Result: {result}")
print(f"Expected: ['0']")
print()

# Test case 4: Show it works correctly with explicit stop values
print("Test 4: Works correctly with explicit stop values")
print("=====================================")
arr = np.array(['hello'], dtype='U')
result = nps.slice(arr, 0, 5)
print(f"nps.slice(np.array(['hello'], dtype='U'), 0, 5)")
print(f"Result: {result}")
print(f"Expected: ['hello']")