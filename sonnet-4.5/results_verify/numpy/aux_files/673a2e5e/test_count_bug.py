import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.count with null character '\\x00'")
print("-" * 60)

test_cases = [
    '',
    'abc',
    'a\x00b',
    '\x00\x00',
]

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_count = nps.count(arr, '\x00')[0]
    py_count = s.count('\x00')
    print(f"count({repr(s):10}, '\\x00'): Python={py_count}, NumPy={np_count}")
    if np_count != py_count:
        print(f"  ^^^ MISMATCH! Expected {py_count}, got {np_count}")