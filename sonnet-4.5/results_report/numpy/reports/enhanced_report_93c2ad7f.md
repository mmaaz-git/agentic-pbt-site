# Bug Report: numpy.strings.count Returns Incorrect Count for Null Characters

**Target**: `numpy.strings.count`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.count()` function returns completely incorrect results when counting null characters (`'\x00'`), treating them as if they match between every character position rather than counting actual null bytes.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=10).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@settings(max_examples=1000)
def test_count_with_bounds(arr, sub, start, end):
    result = nps.count(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].count(sub, start, end)
        assert result[i] == expected, f"Failed for arr[{i}]={repr(arr[i])}, sub={repr(sub)}, start={start}, end={end}: expected {expected}, got {result[i]}"

# Run the test
if __name__ == "__main__":
    # Test with the specific failing example
    arr = np.array(['abc'], dtype=str)
    sub = '\x00'
    start = 0
    end = None

    result = nps.count(arr, sub, start, end)
    expected = arr[0].count(sub, start, end)
    print(f"Testing: arr={repr(arr[0])}, sub={repr(sub)}")
    print(f"Expected: {expected}")
    print(f"Got: {result[0]}")

    if result[0] != expected:
        print(f"\nAssertion Error: Failed for arr[0]={repr(arr[0])}, sub={repr(sub)}, start={start}, end={end}: expected {expected}, got {result[0]}")

    # Run the property-based test
    try:
        test_count_with_bounds()
        print("\nProperty-based test passed!")
    except AssertionError as e:
        print(f"\nProperty-based test failed with: {e}")
```

<details>

<summary>
**Failing input**: `arr = np.array(['abc'], dtype=str), sub = '\x00'`
</summary>
```
Testing: arr=np.str_('abc'), sub='\x00'
Expected: 0
Got: 4

Assertion Error: Failed for arr[0]=np.str_('abc'), sub='\x00', start=0, end=None: expected 0, got 4

Property-based test failed with: Failed for arr[0]=np.str_(''), sub='\x00', start=0, end=None: expected 0, got 1
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

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
```

<details>

<summary>
Incorrect null character counting across multiple test cases
</summary>
```
count(''        , '\x00'): Python=0, NumPy=1
count('abc'     , '\x00'): Python=0, NumPy=4
count('a\x00b'  , '\x00'): Python=1, NumPy=4
count('\x00\x00', '\x00'): Python=2, NumPy=1
```
</details>

## Why This Is A Bug

This violates the expected behavior documented for `numpy.strings.count`, which states it should return "the number of non-overlapping occurrences of substring sub". The function exhibits multiple critical failures:

1. **Fundamentally incorrect counting logic**: For strings without null characters, it returns `len(string) + 1` instead of 0. The string 'abc' has zero null characters but NumPy reports 4.

2. **Inconsistent with Python's str.count()**: The documentation references `str.count` in the "See Also" section, implying similar behavior. Python's `str.count('\x00')` correctly counts null bytes, while NumPy's implementation is completely broken for this case.

3. **Violates mathematical properties of counting**: When counting a single character in a string of length n, the result must be between 0 and n inclusive. NumPy returns n+1 for strings without nulls, which is mathematically impossible.

4. **Unpredictable results with actual nulls**: For strings containing null characters, the results are erratic - '\x00\x00' returns 1 instead of 2, and 'a\x00b' returns 4 instead of 1.

## Relevant Context

The null character (`'\x00'`) is a valid character in both Python strings and NumPy string arrays. It's commonly encountered in:
- Binary file formats
- Network protocols
- C-style string termination
- System programming and low-level data processing

NumPy has documented issues with null character handling in other contexts:
- GitHub Issue #20118: Strings containing a single null byte are incorrectly compared as equal to empty strings
- GitHub Issue #26275: The astype function removes trailing null bytes from strings

The documentation for `numpy.strings.count` does not mention any special handling or limitations for null characters. The function signature accepts any substring without documented restrictions.

This appears to be a systematic implementation error where null characters are being treated as zero-width patterns or string terminators in the underlying C implementation, causing the search algorithm to match at every position between characters plus one additional position.

## Proposed Fix

The issue likely stems from C string handling where null bytes are treated as terminators. The fix requires modifying the underlying implementation to:
1. Use explicit length tracking instead of null-terminated string logic
2. Treat `\x00` as a regular character during pattern matching
3. Ensure the search algorithm doesn't interpret null as a special marker

Without access to the exact C implementation, a high-level fix approach would be:
- Replace any `strlen()` or similar C string functions with explicit length parameters
- Use memory comparison functions like `memcmp()` instead of string comparison functions
- Ensure the pattern matching loop iterates through the full string length, not stopping at null bytes