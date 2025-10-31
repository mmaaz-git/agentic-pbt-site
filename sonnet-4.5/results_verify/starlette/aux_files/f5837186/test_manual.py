import numpy as np
import numpy.strings as nps

print("Manual test cases from bug report:")
print("=" * 60)

test_cases = ['', 'abc', 'a\x00b', '\x00abc']

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_starts = nps.startswith(arr, '\x00')[0]
    py_starts = s.startswith('\x00')
    print(f"startswith({repr(s):10}, '\\x00'): Python={py_starts:5}, NumPy={np_starts:5}, Match={np_starts == py_starts}")

print("\nAdditional test cases:")
print("=" * 60)

# Test with different prefixes to see if the issue is specific to null character
additional_tests = [
    ('abc', 'a'),
    ('abc', 'b'),
    ('abc', '\x00'),
    ('', '\x00'),
    ('\x00', '\x00'),
    ('x\x00y', '\x00'),
    ('x\x00y', 'x'),
]

for s, prefix in additional_tests:
    arr = np.array([s], dtype=str)
    np_starts = nps.startswith(arr, prefix)[0]
    py_starts = s.startswith(prefix)
    print(f"startswith({repr(s):10}, {repr(prefix):5}): Python={py_starts:5}, NumPy={np_starts:5}, Match={np_starts == py_starts}")

# Test with empty prefix
print("\nTest with empty prefix (edge case):")
print("=" * 60)
test_strings = ['', 'abc', 'xyz']
for s in test_strings:
    arr = np.array([s], dtype=str)
    np_starts = nps.startswith(arr, '')[0]
    py_starts = s.startswith('')
    print(f"startswith({repr(s):10}, ''): Python={py_starts:5}, NumPy={np_starts:5}, Match={np_starts == py_starts}")