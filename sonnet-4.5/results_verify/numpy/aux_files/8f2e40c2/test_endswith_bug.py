import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.endswith with null character:")
print("=" * 60)

test_cases = ['', 'abc', 'a\x00b', 'abc\x00']

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_ends = nps.endswith(arr, '\x00')[0]
    py_ends = s.endswith('\x00')

    print(f"String: {repr(s):15}")
    print(f"  Python endswith('\\x00'): {py_ends}")
    print(f"  NumPy endswith('\\x00'):  {np_ends}")
    print(f"  Match: {'✓' if np_ends == py_ends else '✗ MISMATCH'}")
    print()

print("\nAdditional test - checking what NumPy actually stores:")
print("=" * 60)
for s in test_cases:
    arr = np.array([s], dtype=str)
    print(f"Input string: {repr(s):15}")
    print(f"  Stored in NumPy array: {repr(arr[0])}")
    print(f"  Array dtype: {arr.dtype}")
    print()