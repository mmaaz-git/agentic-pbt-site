import numpy as np
import numpy.strings as nps

s = 'hello'
arr = np.array([s])

# Test with concrete values
result1 = nps.slice(arr, 0, 5)
print(f'nps.slice(arr, 0, 5): {repr(result1[0])} (expected: {repr(s[0:5])})')

result2 = nps.slice(arr, 1, 4)
print(f'nps.slice(arr, 1, 4): {repr(result2[0])} (expected: {repr(s[1:4])})')

# Test with None stop
result3 = nps.slice(arr, 0, None)
print(f'nps.slice(arr, 0, None): {repr(result3[0])} (expected: {repr(s[0:None])})')

result4 = nps.slice(arr, 2, None)
print(f'nps.slice(arr, 2, None): {repr(result4[0])} (expected: {repr(s[2:None])})')