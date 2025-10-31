import numpy as np
import numpy.strings as nps

arr = np.array(['hello', 'world', 'test'])

# Test case from bug report
result = nps.slice(arr, 1, None)
print(f"Result with start=1, stop=None: {result}")
print(f"Expected: {np.array(['ello', 'orld', 'est'])}")

# Additional tests to understand the behavior
print("\n--- Additional Tests ---")
result2 = nps.slice(arr, 0, None)
print(f"Result with start=0, stop=None: {result2}")
print(f"Expected (should be full strings): {arr}")

result3 = nps.slice(arr, 2, None)
print(f"Result with start=2, stop=None: {result3}")
print(f"Expected: {np.array(['llo', 'rld', 'st'])}")

# Test without explicitly passing None
result4 = nps.slice(arr, 1)
print(f"\nResult with only start=1 (no stop): {result4}")

# Normal Python slicing for comparison
print(f"\nPython slicing arr[0][1:None]: '{arr[0][1:None]}'")
print(f"Python slicing arr[0][1:]: '{arr[0][1:]}'")