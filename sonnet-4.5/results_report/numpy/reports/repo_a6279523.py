import numpy as np
import numpy.strings as nps

# Test case from the bug report
arr = np.array(['0'])
result = nps.replace(arr, '0', '00')

print(f'Input: {arr}, dtype: {arr.dtype}')
print(f'Result: {result}, dtype: {result.dtype}')
print(f'Expected: ["00"]')
print(f'Actual: {result}')
print()

# What Python's str.replace would do
python_result = '0'.replace('0', '00')
print(f"Python's str.replace('0', '0', '00'): '{python_result}'")
print()

# The assertion that fails
try:
    assert result[0] == '00', f'Expected "00", got "{result[0]}"'
except AssertionError as e:
    print(f'AssertionError: {e}')