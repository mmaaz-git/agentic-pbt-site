import numpy as np
import numpy.strings as nps

print("Testing empty string replacement:")
print("="*50)

# Test with empty string as search pattern
arr = np.array(['abc'], dtype=str)
result = nps.replace(arr, '', 'X')
expected = 'abc'.replace('', 'X')
print(f"Test: Replace empty string with 'X'")
print(f"  Input: {repr(arr[0])}")
print(f"  Replace '' with 'X'")
print(f"  Python's str.replace: {repr(expected)}")
print(f"  numpy.strings.replace: {repr(result[0])}")
print(f"  Match: {result[0] == expected}")