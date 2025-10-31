import numpy as np
import numpy.strings as nps

s = 'hello'
arr = np.array([s])
result = nps.slice(arr, 0, None)

print(f'Expected: {repr(s[0:None])}')
print(f'Got: {repr(result[0])}')
assert result[0] == s[0:None]