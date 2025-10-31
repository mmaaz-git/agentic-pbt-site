import numpy as np
import numpy.strings as nps

# Test the truncation behavior in detail
s = '00'
arr = np.array([s])
print(f"Initial array: {arr}, dtype: {arr.dtype}")

# Test what happens to the 'new' parameter
new_str = 'XXXXXX'
new_arr = np.array(new_str)
print(f"New string: {new_str!r}, as array: {new_arr}, dtype: {new_arr.dtype}")

# What happens when we cast to <U2?
new_cast = new_arr.astype(arr.dtype)
print(f"New string cast to {arr.dtype}: {new_cast!r}")

# Test the actual replacement
result = nps.replace(arr, '0', 'XXXXXX', count=1)
print(f"\nResult: {result[0]!r}, dtype: {result.dtype}")
print(f"Expected: 'XXXXXX0'")
print(f"Python's str.replace result: {s.replace('0', 'XXXXXX', 1)!r}")

# Test with multiple replacements
print("\n=== Testing multiple replacements ===")
arr2 = np.array(['00000'])
result2 = nps.replace(arr2, '0', 'XXXXXX', count=2)
print(f"Input: '00000', replacing first 2 '0's with 'XXXXXX'")
print(f"NumPy result: {result2[0]!r}")
print(f"Python result: {'00000'.replace('0', 'XXXXXX', 2)!r}")

# Test str_len behavior
print("\n=== Testing str_len behavior ===")
from numpy._core.umath import str_len
print(f"str_len(arr): {str_len(arr)}")
print(f"str_len(np.array('XXXXXX')): {str_len(np.array('XXXXXX'))}")
print(f"str_len(np.array('XXXXXX').astype('<U2')): {str_len(np.array('XXXXXX').astype('<U2'))}")