import numpy as np
import numpy.strings as nps

# Test cases demonstrating the bug
test_cases = ['', 'abc', 'a\x00b', 'abc\x00']

print("numpy.strings.endswith() Bug Demonstration")
print("=" * 60)
print()

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_ends = nps.endswith(arr, '\x00')[0]
    py_ends = s.endswith('\x00')

    print(f"String: {repr(s)}")
    print(f"  Python str.endswith('\\x00'): {py_ends}")
    print(f"  NumPy strings.endswith('\\x00'): {np_ends}")
    print(f"  Match: {'✓' if np_ends == py_ends else '✗ MISMATCH'}")
    print()

print("=" * 60)
print("\nConclusion:")
print("numpy.strings.endswith() incorrectly returns True for ALL strings")
print("when checking if they end with null character '\\x00', even when")
print("the null character is not actually present at the end.")