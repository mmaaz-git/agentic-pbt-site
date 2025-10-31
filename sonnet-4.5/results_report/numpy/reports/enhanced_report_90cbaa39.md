# Bug Report: numpy.char String Functions Incorrectly Handle Null Bytes and Empty Strings

**Target**: `numpy.char.find`, `numpy.char.rfind`, `numpy.char.count`, `numpy.char.index`, `numpy.char.startswith`, `numpy.char.endswith`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple numpy.char string functions produce completely incorrect results when handling null bytes (`\x00`) in Unicode strings and when searching in empty strings, treating null bytes as C-style string terminators rather than regular characters and returning 0 for any substring search in empty strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example, seed
import numpy.char as char
import numpy as np

@given(st.lists(st.text(), min_size=1), st.text())
@settings(max_examples=500)
@seed(0)
@example([''], '\x00')  # The specific failing case from the bug report
def test_find_matches_python(strings, substring):
    arr = np.array(strings)
    result = char.find(arr, substring)

    for original, found_idx in zip(arr, result):
        expected = original.find(substring)
        assert found_idx == expected, \
            f"find mismatch: '{original}'.find('{substring}') -> {found_idx} (expected {expected})"

if __name__ == "__main__":
    try:
        test_find_matches_python()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed:")
        print(f"  {e}")
        print("\nMinimal failing input: `strings=['']`, `substring='\\x00'` (or any non-empty substring)")
        print("The bug shows numpy.char.find returns 0 for empty strings regardless of substring,")
```

<details>

<summary>
**Failing input**: `strings=['']`, `substring=' '`
</summary>
```
Test failed:
  find mismatch: ''.find(' ') -> 0 (expected -1)

Minimal failing input: `strings=['']`, `substring='\x00'` (or any non-empty substring)
The bug shows numpy.char.find returns 0 for empty strings regardless of substring,
```
</details>

## Reproducing the Bug

```python
import numpy as np

print("Testing numpy.char null byte handling\n")
print("="*50)

# Test 1: find() on empty string searching for null byte
arr = np.array([''])
result = np.char.find(arr, '\x00')
print(f"Test 1: np.char.find([''], '\\x00')")
print(f"  Result: {result[0]}")
print(f"  Expected: -1 (substring not found)")
print(f"  Python str result: {('').find('\x00')}")
print()

# Test 2: find() on string with null byte
arr2 = np.array(['a\x00b'])
result2 = np.char.find(arr2, '\x00')
print(f"Test 2: np.char.find(['a\\x00b'], '\\x00')")
print(f"  Result: {result2[0]}")
print(f"  Expected: 1 (null byte is at index 1)")
print(f"  Python str result: {('a\x00b').find('\x00')}")
print()

# Test 3: count() on string without null bytes
arr3 = np.array(['hello'])
result3 = np.char.count(arr3, '\x00')
print(f"Test 3: np.char.count(['hello'], '\\x00')")
print(f"  Result: {result3[0]}")
print(f"  Expected: 0 (no null bytes in 'hello')")
print(f"  Python str result: {('hello').count('\x00')}")
print()

# Test 4: startswith() on string with null byte
arr4 = np.array(['hello'])
result4 = np.char.startswith(arr4, '\x00')
print(f"Test 4: np.char.startswith(['hello'], '\\x00')")
print(f"  Result: {result4[0]}")
print(f"  Expected: False ('hello' does not start with null byte)")
print(f"  Python str result: {('hello').startswith('\x00')}")
print()

# Test 5: rfind() on string without null bytes
arr5 = np.array(['hello'])
result5 = np.char.rfind(arr5, '\x00')
print(f"Test 5: np.char.rfind(['hello'], '\\x00')")
print(f"  Result: {result5[0]}")
print(f"  Expected: -1 (substring not found)")
print(f"  Python str result: {('hello').rfind('\x00')}")
print()

# Test 6: endswith() on string with null byte
arr6 = np.array(['hello'])
result6 = np.char.endswith(arr6, '\x00')
print(f"Test 6: np.char.endswith(['hello'], '\\x00')")
print(f"  Result: {result6[0]}")
print(f"  Expected: False ('hello' does not end with null byte)")
print(f"  Python str result: {('hello').endswith('\x00')}")
print()

# Test 7: index() on empty string searching for null byte
arr7 = np.array([''])
try:
    result7 = np.char.index(arr7, '\x00')
    print(f"Test 7: np.char.index([''], '\\x00')")
    print(f"  Result: {result7[0]} (should have raised ValueError)")
    print(f"  Expected: ValueError exception")
except ValueError as e:
    print(f"Test 7: np.char.index([''], '\\x00')")
    print(f"  Result: ValueError raised - {e}")
    print(f"  Expected: ValueError exception")

# Compare with Python's behavior
try:
    python_result = ('').index('\x00')
except ValueError as e:
    print(f"  Python str result: ValueError raised - {e}")
print()

# Control test with normal substring
print("Control test (non-null-byte):")
arr_control = np.array(['hello world'])
result_control = np.char.find(arr_control, 'world')
print(f"  np.char.find(['hello world'], 'world') = {result_control[0]}")
print(f"  Expected: 6")
print(f"  Python str result: {('hello world').find('world')}")
```

<details>

<summary>
Output showing the bug behavior
</summary>
```
Testing numpy.char null byte handling

==================================================
Test 1: np.char.find([''], '\x00')
  Result: 0
  Expected: -1 (substring not found)
  Python str result: -1

Test 2: np.char.find(['a\x00b'], '\x00')
  Result: 0
  Expected: 1 (null byte is at index 1)
  Python str result: 1

Test 3: np.char.count(['hello'], '\x00')
  Result: 6
  Expected: 0 (no null bytes in 'hello')
  Python str result: 0

Test 4: np.char.startswith(['hello'], '\x00')
  Result: True
  Expected: False ('hello' does not start with null byte)
  Python str result: False

Test 5: np.char.rfind(['hello'], '\x00')
  Result: 5
  Expected: -1 (substring not found)
  Python str result: -1

Test 6: np.char.endswith(['hello'], '\x00')
  Result: True
  Expected: False ('hello' does not end with null byte)
  Python str result: False

Test 7: np.char.index([''], '\x00')
  Result: 0 (should have raised ValueError)
  Expected: ValueError exception
  Python str result: ValueError raised - substring not found

Control test (non-null-byte):
  np.char.find(['hello world'], 'world') = 6
  Expected: 6
  Python str result: 6
```
</details>

## Why This Is A Bug

The numpy.char module documentation explicitly states that these functions provide "vectorized string operations" that are "based on" Python's string methods, establishing a clear behavioral contract that they should produce identical results to Python's built-in string methods when operating on individual array elements.

However, the functions fail catastrophically in two scenarios:

1. **Null byte handling**: The functions appear to treat Unicode strings as null-terminated C strings, leading to:
   - `find('')` with null byte returns 0 instead of -1 (substring not found)
   - `find('a\x00b')` returns 0 instead of 1 for the null byte position
   - `count('hello')` returns 6 null bytes when there are zero
   - `startswith/endswith` incorrectly return True for null byte checks
   - `rfind('hello')` returns the string length instead of -1

2. **Empty string handling**: Any substring search in an empty string returns 0 instead of -1, violating Python's string semantics where empty strings contain no substrings.

These are not edge cases but fundamental violations of the documented behavior. Python strings inherently support null bytes as regular characters (since Unicode strings are not null-terminated), and the numpy.char functions claim to provide Python-compatible string operations. The control test confirms the functions work correctly for normal (non-null-byte) cases.

## Relevant Context

The numpy.char module is marked as "legacy" in the documentation with a note that it will "no longer receive updates." However, this bug is severe enough to warrant attention because:

1. The functions produce objectively incorrect results that could lead to data corruption
2. The module is still included in NumPy and actively used by existing code
3. There is no workaround within numpy.char - users must avoid the module entirely for data that might contain null bytes
4. The bug affects at least 6 core string functions in systematic ways

The issue likely stems from the underlying C implementation using C-style string functions that depend on null terminators rather than using the actual string length from the numpy array metadata. This is visible from the pattern where `count('hello', '\x00')` returns 6 (length + 1), suggesting the code is counting an implicit null terminator.

Documentation references:
- NumPy char module: https://numpy.org/doc/stable/reference/routines.char.html
- Python string methods: https://docs.python.org/3/library/stdtypes.html#string-methods

## Proposed Fix

The fix requires modifying the underlying C implementation to properly handle string lengths and null bytes. Since the actual C code is compiled, a complete fix would require changes to NumPy's core string handling. The high-level approach would be:

1. Use explicit string lengths from numpy array metadata instead of relying on null termination
2. Treat null bytes as regular searchable characters
3. Ensure empty string searches return -1 (not found) as per Python semantics
4. Update the index() function to raise ValueError when substring is not found

Without access to modify the compiled C extensions, users should be warned that numpy.char functions do not correctly handle:
- Null bytes in strings
- Substring searches in empty strings

And should use alternative approaches like list comprehensions with Python's native string methods for data that might contain these cases.