import numpy.char as char
import numpy as np

# Test with explicitly sized arrays
test_cases = [
    (np.array('a', dtype='<U1'), 'a', 'aa'),  # U1 -> should truncate
    (np.array('a', dtype='<U2'), 'a', 'aa'),  # U2 -> should work
    (np.array('a', dtype='<U10'), 'a', 'aa'), # U10 -> should work
]

for arr, old, new in test_cases:
    result = char.replace(arr, old, new)
    print(f"Input dtype: {arr.dtype}, Input: {repr(arr.item())}")
    print(f"  Replacing {repr(old)} with {repr(new)}")
    print(f"  Result dtype: {result.dtype}, Result: {repr(result.item())}")
    print(f"  Expected: {repr(arr.item().replace(old, new))}")
    print()