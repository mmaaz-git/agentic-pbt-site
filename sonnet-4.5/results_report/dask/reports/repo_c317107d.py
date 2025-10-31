import numpy as np
import numpy.strings as nps

# Minimal failing test case
s = '0'
arr = np.array([s])
result = nps.replace(arr, '0', '00')

print(f"Input string: '{s}'")
print(f"Input array dtype: {arr.dtype}")
print(f"Replacement: '0' -> '00'")
print(f"Expected result: '00'")
print(f"Actual result: '{result[0]}'")
print(f"Result array dtype: {result.dtype}")
print()

# Verify the bug
try:
    assert result[0] == '00', f"Expected '00', but got '{result[0]}'"
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")