# Bug Report: numpy.strings.rfind Returns String Length Instead of -1 for Non-Existent Substrings Including Null Characters

**Target**: `numpy.strings.rfind`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.rfind()` function returns the string's length instead of -1 when searching for substrings that don't exist, particularly null characters (`'\x00'`), and returns incorrect indices when null characters are present in the string.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

string_arrays = st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str))

@given(string_arrays, st.text(min_size=1, max_size=10))
@settings(max_examples=1000)
def test_rfind_consistency(arr, sub):
    result = nps.rfind(arr, sub)
    for i in range(len(arr)):
        expected = arr[i].rfind(sub)
        assert result[i] == expected, f"Failed on arr[{i}]='{arr[i]}', sub='{sub}': expected {expected}, got {result[i]}"

if __name__ == "__main__":
    # Run the test
    try:
        test_rfind_consistency()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nTrying specific failing case:")
        arr = np.array(['abc'], dtype=str)
        sub = '\x00'
        result = nps.rfind(arr, sub)
        expected = arr[0].rfind(sub)
        print(f"arr = np.array(['abc'], dtype=str)")
        print(f"sub = '\\x00'")
        print(f"Python str.rfind: {expected}")
        print(f"NumPy rfind: {result[0]}")
        print(f"Expected: -1, Got: {result[0]}")
```

<details>

<summary>
**Failing input**: `arr = np.array([''], dtype=str), sub = ' '`
</summary>
```
Test failed: Failed on arr[0]='', sub=' ': expected -1, got 0

Trying specific failing case:
arr = np.array(['abc'], dtype=str)
sub = '\x00'
Python str.rfind: -1
NumPy rfind: 3
Expected: -1, Got: 3
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

# Test various cases with null character searching
test_cases = [
    '',
    'abc',
    'a\x00b',
    '\x00\x00',
    'hello world',
    '\x00',
    'a',
    'abc\x00',
    '\x00abc',
    'a\x00b\x00c'
]

print("Testing numpy.strings.rfind with null character ('\\x00'):")
print("=" * 70)
print(f"{'String':<20} | {'Python rfind':<15} | {'NumPy rfind':<15} | {'Match?':<10}")
print("-" * 70)

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_rfind = nps.rfind(arr, '\x00')[0]
    py_rfind = s.rfind('\x00')
    match = "✓" if np_rfind == py_rfind else "✗"

    # Format string representation for display
    s_repr = repr(s) if s else "''"
    print(f"{s_repr:<20} | {py_rfind:<15} | {np_rfind:<15} | {match:<10}")

print("\n" + "=" * 70)
print("\nKey observations:")
print("1. When '\\x00' is NOT in string: NumPy returns len(string) instead of -1")
print("2. When '\\x00' IS in string: NumPy often returns wrong position")
print("3. Python's str.rfind correctly returns -1 when not found")
```

<details>

<summary>
NumPy rfind systematically returns incorrect values for null character searches
</summary>
```
Testing numpy.strings.rfind with null character ('\x00'):
======================================================================
String               | Python rfind    | NumPy rfind     | Match?
----------------------------------------------------------------------
''                   | -1              | 0               | ✗
'abc'                | -1              | 3               | ✗
'a\x00b'             | 1               | 3               | ✗
'\x00\x00'           | 1               | 0               | ✗
'hello world'        | -1              | 11              | ✗
'\x00'               | 0               | 0               | ✓
'a'                  | -1              | 1               | ✗
'abc\x00'            | 3               | 3               | ✓
'\x00abc'            | 0               | 4               | ✗
'a\x00b\x00c'        | 3               | 5               | ✗

======================================================================

Key observations:
1. When '\x00' is NOT in string: NumPy returns len(string) instead of -1
2. When '\x00' IS in string: NumPy often returns wrong position
3. Python's str.rfind correctly returns -1 when not found
```
</details>

## Why This Is A Bug

This violates the documented behavior of `numpy.strings.rfind` in multiple critical ways:

1. **Contract Violation**: The NumPy documentation explicitly states that rfind should return -1 when a substring is not found. Instead, it returns the string's length when searching for null characters that don't exist.

2. **Python Incompatibility**: NumPy's string functions are designed to provide array-based equivalents of Python's string methods. Python's `str.rfind('\x00')` correctly returns -1 when the null character is not found, while NumPy returns the string length.

3. **Incorrect Position Returns**: Even when null characters ARE present in the string, the function often returns wrong positions. For example, `'a\x00b'.rfind('\x00')` should return 1 but NumPy returns 3.

4. **Pattern Analysis**: The bug shows a clear pattern - when null characters are not found, NumPy consistently returns `len(string)` instead of -1, suggesting a systematic implementation error in handling null character searches.

5. **Data Integrity Risk**: This bug could cause serious data processing errors in production systems that rely on correct substring position detection, potentially leading to out-of-bounds access or incorrect data manipulation.

## Relevant Context

The bug affects not just null characters but appears to be a broader issue with how NumPy handles certain substring searches. The Hypothesis test caught the issue even with a regular space character on an empty string (returning 0 instead of -1).

Documentation references:
- NumPy strings.rfind: https://numpy.org/doc/stable/reference/generated/numpy.strings.rfind.html
- Python str.rfind: https://docs.python.org/3/library/stdtypes.html#str.rfind

The issue appears to be in the underlying C implementation of string searching in NumPy, where null characters or empty searches may be incorrectly handled, possibly due to C string termination conventions interfering with the search logic.

## Proposed Fix

The bug likely resides in the C extension code that implements string searching. Without access to the exact C implementation, a high-level fix would involve:

1. Properly handling null characters as searchable characters rather than string terminators
2. Ensuring that when a substring is not found, -1 is returned consistently regardless of the substring content
3. Correcting the index calculation when null characters are present in the string

The implementation should treat null characters (`\x00`) as regular characters in the search algorithm and ensure that the "not found" return value is always -1, not the string length. This may require changes to how the underlying C code handles string boundaries and null termination.