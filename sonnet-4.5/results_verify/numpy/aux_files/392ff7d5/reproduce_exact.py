import numpy as np
import numpy.strings as nps

arr = np.array(['0'])
result = nps.replace(arr, '0', '00')

print(f'Input: {arr}, dtype: {arr.dtype}')
print(f'Result: {result}, dtype: {result.dtype}')
print(f'Expected: ["00"]')
print(f'Actual: {result}')

assert result[0] == '00', f'Expected "00", got {result[0]}'