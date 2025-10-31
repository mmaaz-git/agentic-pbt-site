import numpy as np
import numpy.strings as nps

# Test case 1: slice(arr, 1, None) should slice from index 1 to end
arr = np.array(['hello', 'world', 'test'])

result = nps.slice(arr, 1, None)
print(f"Test 1: nps.slice(['hello', 'world', 'test'], 1, None)")
print(f"Result:   {result}")
print(f"Expected: {np.array(['ello', 'orld', 'est'])}")
print()

# Test case 2: slice(arr, 0, None) should return full strings
arr2 = np.array(['hello'])
result2 = nps.slice(arr2, 0, None)
print(f"Test 2: nps.slice(['hello'], 0, None)")
print(f"Result:   {result2}")
print(f"Expected: {np.array(['hello'])}")
print()

# Test case 3: slice(arr, 2, None) should slice from index 2 to end
arr3 = np.array(['abcdef', 'ghijkl'])
result3 = nps.slice(arr3, 2, None)
print(f"Test 3: nps.slice(['abcdef', 'ghijkl'], 2, None)")
print(f"Result:   {result3}")
print(f"Expected: {np.array(['cdef', 'ijkl'])}")
print()

# Comparison with Python slicing behavior
print("Comparison with standard Python slicing:")
s = 'hello'
print(f"Python 'hello'[1:None] = '{s[1:None]}'")
print(f"Python 'hello'[1:]     = '{s[1:]}'")
print(f"numpy.strings.slice(['hello'], 1, None) = {nps.slice(np.array(['hello']), 1, None)}")