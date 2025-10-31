import numpy as np
import numpy.strings as nps

s = '00'
arr = np.array([s])
result = nps.replace(arr, '0', 'XXXXXX', count=1)

print(f"Input: {s!r}")
print(f"Python result: {s.replace('0', 'XXXXXX', 1)!r}")
print(f"NumPy result:  {result[0]!r}")

assert result[0] == 'XXXXXX0', f"Expected 'XXXXXX0', got {result[0]!r}"