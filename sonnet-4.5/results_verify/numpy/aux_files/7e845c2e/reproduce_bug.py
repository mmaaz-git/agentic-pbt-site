import numpy as np
import numpy.strings as ns

print("Bug 1: str_len")
arr = np.array(['\x00'])
print(f"  str_len(['\\x00']): {ns.str_len(arr)[0]} (expected: 1)")

arr = np.array(['a\x00'])
print(f"  str_len(['a\\x00']): {ns.str_len(arr)[0]} (expected: 2)")

print("\nBug 2: capitalize")
arr = np.array(['\x00'])
result = ns.capitalize(arr)
print(f"  capitalize(['\\x00']): {repr(result[0])} (expected: '\\x00')")

print("\nBug 3: find")
arr = np.array([''])
result = ns.find(arr, '\x00')
print(f"  find([''], '\\x00'): {result[0]} (expected: -1)")

arr = np.array(['abc'])
result = ns.find(arr, '\x00')
print(f"  find(['abc'], '\\x00'): {result[0]} (expected: -1)")

print("\nBug 4: slice")
arr = np.array(['\x000'])
result = ns.slice(arr, 0, 1)
print(f"  slice(['\\x000'], 0, 1): {repr(result[0])} (expected: '\\x00')")

arr = np.array(['a\x00b'])
result = ns.slice(arr, 0, 2)
print(f"  slice(['a\\x00b'], 0, 2): {repr(result[0])} (expected: 'a\\x00')")

print("\n\nPython standard behavior for comparison:")
print(f"len('\\x00'): {len('\x00')}")
print(f"'\\x00'.capitalize(): {repr('\x00'.capitalize())}")
print(f"''.find('\\x00'): {''.find('\x00')}")
print(f"'abc'.find('\\x00'): {'abc'.find('\x00')}")
print(f"'\\x000'[0:1]: {repr('\x000'[0:1])}")
print(f"'a\\x00b'[0:2]: {repr('a\x00b'[0:2])}")