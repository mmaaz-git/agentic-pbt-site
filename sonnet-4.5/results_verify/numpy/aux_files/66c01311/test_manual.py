import numpy as np
import numpy.strings as nps

arr1 = np.array([''], dtype='<U100')
arr2 = np.array(['\x00'], dtype='<U100')

print(f'arr1[0]: {repr(arr1[0])}')
print(f'arr2[0]: {repr(arr2[0])}')

result = nps.add(arr1, arr2)
print(f'nps.add result: {repr(result[0])}')

expected = '' + '\x00'
print(f'Python result: {repr(expected)}')
print(f'Match: {result[0] == expected}')

arr_null = np.array(['\x00'], dtype='<U100')
print(f'\nDirect null array: {repr(arr_null[0])}')
print(f'Length: {len(arr_null[0])}')