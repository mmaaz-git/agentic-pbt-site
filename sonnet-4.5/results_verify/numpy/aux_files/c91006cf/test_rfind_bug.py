import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.rfind with null characters:")
print("=" * 60)

test_cases = ['', 'abc', 'a\x00b', '\x00\x00']

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_rfind = nps.rfind(arr, '\x00')[0]
    py_rfind = s.rfind('\x00')

    match = "✓" if np_rfind == py_rfind else "✗"
    print(f"rfind({repr(s):10}, '\\x00'): Python={py_rfind:3}, NumPy={np_rfind:3} {match}")

print("\n" + "=" * 60)
print("Testing with other substrings for comparison:")
print("=" * 60)

# Test with regular characters
test_string = "hello world"
arr = np.array([test_string], dtype=str)

for sub in ['o', 'l', 'world', 'xyz']:
    np_rfind = nps.rfind(arr, sub)[0]
    py_rfind = test_string.rfind(sub)
    match = "✓" if np_rfind == py_rfind else "✗"
    print(f"rfind({repr(test_string)}, {repr(sub):7}): Python={py_rfind:3}, NumPy={np_rfind:3} {match}")