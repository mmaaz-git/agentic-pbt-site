# Bug Report: numpy.strings.endswith and rfind Incorrect Behavior with Null Characters

**Target**: `numpy.strings.endswith` and `numpy.strings.rfind`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.endswith` function incorrectly returns `True` when checking if any string (including empty strings) ends with `'\x00'`, and `numpy.strings.rfind` returns incorrect indices instead of `-1` when searching for null characters in strings where they don't exist.

## Property-Based Test

```python
import numpy as np
import numpy.strings
from hypothesis import given, strategies as st, settings, example


@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
@example([''], '\x00')  # Force the failing case
@settings(max_examples=10)
def test_endswith_rfind_consistency(strings, suffix):
    arr = np.array(strings)
    endswith_result = numpy.strings.endswith(arr, suffix)
    rfind_result = numpy.strings.rfind(arr, suffix)
    str_lens = numpy.strings.str_len(arr)

    for ew, rfind_idx, s_len in zip(endswith_result, rfind_result, str_lens):
        if ew:
            expected_idx = s_len - len(suffix)
            assert rfind_idx == expected_idx, f"endswith=True but rfind={rfind_idx} != expected {expected_idx} (str_len={s_len}, suffix_len={len(suffix)})"

# Run the test
if __name__ == "__main__":
    test_endswith_rfind_consistency()
```

<details>

<summary>
**Failing input**: `strings=[''], suffix='\x00'`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 20, in <module>
    test_endswith_rfind_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "<string>", line 7, in test_endswith_rfind_consistency
    @example([''], '\x00')  # Force the failing case
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "<string>", line 18, in test_endswith_rfind_consistency
    assert rfind_idx == expected_idx, f'endswith=True but rfind={rfind_idx} != expected {expected_idx} (str_len={s_len}, suffix_len={len(suffix)})'
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: endswith=True but rfind=0 != expected -1 (str_len=0, suffix_len=1)
Falsifying explicit example: test_endswith_rfind_consistency(
    strings=[''],
    suffix='\x00',
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings

# Test case: empty string with null character
arr = np.array([''])

# NumPy results
endswith_result = numpy.strings.endswith(arr, '\x00')
rfind_result = numpy.strings.rfind(arr, '\x00')

print(f"NumPy endswith([''], '\\x00'): {endswith_result[0]}")
print(f"NumPy rfind([''], '\\x00'):    {rfind_result[0]}")
print()

# Python results for comparison
print(f"Python ''.endswith('\\x00'): {repr(''.endswith('\x00'))}")
print(f"Python ''.rfind('\\x00'):    {repr(''.rfind('\x00'))}")
print()

# Additional tests to understand the behavior
print("Additional test cases:")
print(f"NumPy endswith(['a'], '\\x00'): {numpy.strings.endswith(np.array(['a']), '\x00')[0]}")
print(f"NumPy rfind(['a'], '\\x00'):    {numpy.strings.rfind(np.array(['a']), '\x00')[0]}")
print(f"Python 'a'.endswith('\\x00'): {repr('a'.endswith('\x00'))}")
print(f"Python 'a'.rfind('\\x00'):    {repr('a'.rfind('\x00'))}")
```

<details>

<summary>
NumPy incorrectly reports that strings end with null characters
</summary>
```
NumPy endswith([''], '\x00'): True
NumPy rfind([''], '\x00'):    0

Python ''.endswith('\x00'): False
Python ''.rfind('\x00'):    -1

Additional test cases:
NumPy endswith(['a'], '\x00'): True
NumPy rfind(['a'], '\x00'):    1
Python 'a'.endswith('\x00'): False
Python 'a'.rfind('\x00'):    -1
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Semantic Violation**: An empty string cannot logically end with any character, including the null character `'\x00'`. Python's `str.endswith('')` correctly returns `False`, but NumPy returns `True`.

2. **Documentation Contradiction**: Both `numpy.strings.endswith` and `numpy.strings.rfind` explicitly reference Python's `str.endswith` and `str.rfind` in their "See Also" sections, establishing that they should follow Python's string semantics. The NumPy documentation for `rfind` specifically states it returns `-1` when the substring is not found.

3. **Internal Inconsistency**: The property test reveals that when `endswith` returns `True`, `rfind` should return the position where the suffix starts (string length minus suffix length). For an empty string with a one-character suffix, this would be -1, but NumPy's `rfind` returns 0, creating an impossible situation where a substring is found at position 0 in an empty string.

4. **Broader Pattern**: The bug affects not just empty strings but all strings - NumPy incorrectly reports that ALL strings end with `'\x00'`, which is fundamentally wrong.

## Relevant Context

The issue likely stems from NumPy's internal C-style string handling where null characters (`\x00`) are used as string terminators. This implementation detail appears to be leaking into the Python API, causing the functions to incorrectly identify the null terminator as part of the string content.

Key documentation references:
- `numpy.strings.endswith`: States "Returns a boolean array which is `True` where the string element in `a` ends with `suffix`" and references `str.endswith`
- `numpy.strings.rfind`: States "return the highest index in the string where substring `sub` is found" and should return -1 when not found, referencing `str.rfind`

Both functions are documented as element-wise applications of the corresponding Python string methods, meaning they should produce identical results when called on individual strings.

## Proposed Fix

The functions need to properly handle null character searches by distinguishing between:
1. Null characters that are part of the string content
2. Null terminators used internally for C-style string storage

A high-level fix approach would be:
1. When searching for `'\x00'` as a suffix or substring, the functions should check against the actual string content, not including any internal null terminators
2. Empty strings should always return `False` for `endswith` with any non-empty suffix
3. `rfind` should return `-1` when the substring is genuinely not found in the string content

The fix would likely involve modifying the underlying C implementation to properly handle the boundary between string content and null terminators, ensuring that the Python API behavior matches Python's built-in string methods exactly.