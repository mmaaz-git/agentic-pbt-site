import numpy as np
import numpy.strings as nps

print("Comprehensive null character testing:")
print("=" * 70)

# Test various strings with and without null characters
test_data = [
    ("", '\x00', "Empty string with null search"),
    ("abc", '\x00', "Regular string with null search"),
    ('\x00', '\x00', "Single null searching for null"),
    ('a\x00b', '\x00', "String with null in middle"),
    ('\x00\x00', '\x00', "Double null searching for null"),
    ('abc\x00', '\x00', "String ending with null"),
    ('\x00abc', '\x00', "String starting with null"),
    ('a\x00b\x00c', '\x00', "String with multiple nulls"),
]

for s, sub, description in test_data:
    arr = np.array([s], dtype=str)
    np_rfind = nps.rfind(arr, sub)[0]
    py_rfind = s.rfind(sub)

    match = "✓" if np_rfind == py_rfind else "✗"
    print(f"{description:35} | str={repr(s):15} | Python={py_rfind:3} | NumPy={np_rfind:3} {match}")

print("\n" + "=" * 70)
print("Testing numpy.strings.find (not rfind) for comparison:")
print("=" * 70)

for s, sub, description in test_data:
    arr = np.array([s], dtype=str)
    np_find = nps.find(arr, sub)[0]
    py_find = s.find(sub)

    match = "✓" if np_find == py_find else "✗"
    print(f"{description:35} | str={repr(s):15} | Python={py_find:3} | NumPy={np_find:3} {match}")

print("\n" + "=" * 70)
print("Cross-checking: Does len(s) == NumPy rfind result when null not found?")
print("=" * 70)

for s in ["", "a", "ab", "abc", "abcd", "hello world"]:
    arr = np.array([s], dtype=str)
    np_rfind = nps.rfind(arr, '\x00')[0]
    py_rfind = s.rfind('\x00')
    string_len = len(s)

    print(f"str={repr(s):15} | len={string_len:2} | np.rfind={np_rfind:3} | Matches len? {np_rfind == string_len}")