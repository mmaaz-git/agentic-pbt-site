import numpy as np
import numpy.strings as nps

print("Testing numpy.strings upper/lower with null characters")
print("=" * 60)

test_cases = ['\x00', '\x00\x00', 'a\x00b', 'hello\x00world']

for test_str in test_cases:
    arr = np.array([test_str])
    py_upper = test_str.upper()
    np_upper = nps.upper(arr)[0]
    py_lower = test_str.lower()
    np_lower = nps.lower(arr)[0]

    print(f"Input: {repr(test_str)}")
    print(f"  upper() - Python: {repr(py_upper)}, NumPy: {repr(np_upper)}")
    print(f"  lower() - Python: {repr(py_lower)}, NumPy: {repr(np_lower)}")
    print()