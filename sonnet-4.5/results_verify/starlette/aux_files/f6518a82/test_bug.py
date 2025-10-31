import numpy as np
import numpy.strings as nps

# Simple reproduction case
print("=== Simple Reproduction Case ===")
s = '00'
arr = np.array([s])
result = nps.replace(arr, '0', 'XXXXXX', count=1)

print(f"Input: {s!r}")
print(f"Python result: {s.replace('0', 'XXXXXX', 1)!r}")
print(f"NumPy result:  {result[0]!r}")
print(f"Expected: 'XXXXXX0'")
print(f"Got: {result[0]!r}")
print(f"Match: {result[0] == 'XXXXXX0'}")