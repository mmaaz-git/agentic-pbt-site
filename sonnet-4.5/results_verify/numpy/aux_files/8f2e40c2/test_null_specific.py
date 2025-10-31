import numpy as np
import numpy.strings as nps

print("Testing endswith with various null character scenarios:")
print("=" * 60)

# Test 1: Direct test with null suffix
test_strings = ['', 'a', 'ab', 'abc', 'abcd', 'hello', 'world']
print("\nTest 1: Various strings ending with '\\x00'")
for s in test_strings:
    arr = np.array([s], dtype=str)
    np_result = nps.endswith(arr, '\x00')[0]
    py_result = s.endswith('\x00')
    print(f"  {repr(s):10} -> NumPy: {np_result}, Python: {py_result}, Match: {np_result == py_result}")

# Test 2: Strings that actually end with null
print("\nTest 2: Strings that DO end with '\\x00'")
null_ending = ['a\x00', 'hello\x00', '\x00']
for s in null_ending:
    arr = np.array([s], dtype=str)
    np_result = nps.endswith(arr, '\x00')[0]
    py_result = s.endswith('\x00')
    print(f"  {repr(s):10} -> NumPy: {np_result}, Python: {py_result}, Match: {np_result == py_result}")

# Test 3: Check how NumPy stores these strings
print("\nTest 3: How NumPy stores strings with nulls")
test_with_nulls = ['abc', 'abc\x00', 'a\x00bc', '\x00abc']
for s in test_with_nulls:
    arr = np.array([s], dtype=str)
    print(f"  Input: {repr(s):10} -> Stored as: {repr(arr[0]):10} (dtype: {arr.dtype})")

# Test 4: Test with other suffixes for comparison
print("\nTest 4: Testing with non-null suffixes (for comparison)")
arr = np.array(['abc', 'hello', 'world'], dtype=str)
suffixes = ['c', 'o', 'd', 'bc', 'llo']
for suffix in suffixes:
    np_results = nps.endswith(arr, suffix)
    py_results = [s.endswith(suffix) for s in ['abc', 'hello', 'world']]
    matches = all(np_results[i] == py_results[i] for i in range(len(arr)))
    print(f"  Suffix {repr(suffix):5} -> All match: {matches}")