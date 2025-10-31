# Bug Report: numpy.strings upper/lower Return Empty String for Null-Only Strings

**Target**: `numpy.strings.upper`, `numpy.strings.lower`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.upper()` and `numpy.strings.lower()` incorrectly return an empty string for strings consisting only of null characters (`\x00`), instead of preserving the null characters as Python's `str.upper()` and `str.lower()` do.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(), min_size=1, max_size=10))
def test_upper_matches_python(strings):
    """Test that NumPy's upper() matches Python's str.upper() element-wise."""
    for s in strings:
        arr = np.array([s])
        np_result = nps.upper(arr)[0]
        py_result = s.upper()
        assert np_result == py_result, f"Mismatch for {repr(s)}: NumPy returned {repr(np_result)}, Python returned {repr(py_result)}"

@given(st.lists(st.text(), min_size=1, max_size=10))
def test_lower_matches_python(strings):
    """Test that NumPy's lower() matches Python's str.lower() element-wise."""
    for s in strings:
        arr = np.array([s])
        np_result = nps.lower(arr)[0]
        py_result = s.lower()
        assert np_result == py_result, f"Mismatch for {repr(s)}: NumPy returned {repr(np_result)}, Python returned {repr(py_result)}"
```

<details>

<summary>
**Failing input**: `['\x00']`
</summary>
```
Running Hypothesis property-based tests for NumPy strings upper/lower...
======================================================================

Testing with known failing case: ['\x00']
✗ upper() test failed: Mismatch for '\x00': NumPy returned np.str_(''), Python returned '\x00'
✗ lower() test failed: Mismatch for '\x00': NumPy returned np.str_(''), Python returned '\x00'

Running general Hypothesis tests...
----------------------------------------
Found failing case for upper(): ['ﬀ']
Found failing case for lower(): ['\x00']
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Reproduce the NumPy strings upper/lower null character bug."""

import numpy as np
import numpy.strings as nps

# Test cases including null-only strings and mixed strings
test_cases = ['\x00', '\x00\x00', 'a\x00b', 'hello\x00world']

print("Bug Reproduction: numpy.strings upper/lower with null characters")
print("=" * 65)
print()

for test_str in test_cases:
    # Create NumPy array
    arr = np.array([test_str])

    # Test upper()
    py_upper = test_str.upper()
    np_upper = nps.upper(arr)[0]

    # Test lower()
    py_lower = test_str.lower()
    np_lower = nps.lower(arr)[0]

    print(f"Input: {repr(test_str)}")
    print(f"  upper() - Python: {repr(py_upper)}, NumPy: {repr(np_upper)}")
    print(f"  lower() - Python: {repr(py_lower)}, NumPy: {repr(np_lower)}")

    # Check if results match
    upper_match = py_upper == np_upper
    lower_match = py_lower == np_lower

    if not upper_match or not lower_match:
        print(f"  ❌ MISMATCH DETECTED!")
    else:
        print(f"  ✓ Results match")
    print()

print("Summary:")
print("-" * 30)
print("For null-only strings, NumPy returns empty string instead of preserving the nulls.")
print("This violates the documented behavior of calling str methods element-wise.")
```

<details>

<summary>
NumPy incorrectly converts null-only strings to empty strings
</summary>
```
Bug Reproduction: numpy.strings upper/lower with null characters
=================================================================

Input: '\x00'
  upper() - Python: '\x00', NumPy: np.str_('')
  lower() - Python: '\x00', NumPy: np.str_('')
  ❌ MISMATCH DETECTED!

Input: '\x00\x00'
  upper() - Python: '\x00\x00', NumPy: np.str_('')
  lower() - Python: '\x00\x00', NumPy: np.str_('')
  ❌ MISMATCH DETECTED!

Input: 'a\x00b'
  upper() - Python: 'A\x00B', NumPy: np.str_('A\x00B')
  lower() - Python: 'a\x00b', NumPy: np.str_('a\x00b')
  ✓ Results match

Input: 'hello\x00world'
  upper() - Python: 'HELLO\x00WORLD', NumPy: np.str_('HELLO\x00WORLD')
  lower() - Python: 'hello\x00world', NumPy: np.str_('hello\x00world')
  ✓ Results match

Summary:
------------------------------
For null-only strings, NumPy returns empty string instead of preserving the nulls.
This violates the documented behavior of calling str methods element-wise.
```
</details>

## Why This Is A Bug

This is a clear violation of NumPy's documented behavior. The documentation for `numpy.strings.upper()` and `numpy.strings.lower()` explicitly states that these functions "Call str.upper/lower element-wise". However:

1. **Contract Violation**: Python's `str.upper('\x00')` returns `'\x00'` (unchanged, as null is not a cased character), but NumPy returns an empty string `''`.

2. **Data Loss**: The null character (`\x00`, Unicode U+0000) is a valid character that should be preserved. Converting it to an empty string constitutes data loss.

3. **Inconsistent Behavior**: The bug only manifests for strings consisting entirely of null characters. Mixed strings like `'a\x00b'` work correctly, preserving the null in the middle.

4. **Silent Failure**: The function doesn't raise an error or warning - it silently returns incorrect results, which could lead to hard-to-debug issues in production code.

The null character is not an obscure edge case - it appears in:
- Binary data processing
- C string interoperability
- Network protocols
- File format parsing
- Database BLOB handling

## Relevant Context

The bug appears to stem from the underlying C implementation (`_vec_string` from `numpy._core.multiarray`) likely treating `\x00` as a string terminator in C-style string handling. When a string consists only of null characters, the C code may interpret this as an empty string due to null-termination conventions.

The implementation is found in:
- Python wrapper: `/numpy/_core/strings.py:1102-1173`
- Uses `_vec_string` from `numpy._core.multiarray` (C extension)
- Documentation: https://numpy.org/doc/stable/reference/generated/numpy.char.upper.html

Interestingly, the Hypothesis test also revealed that `upper()` has issues with the ligature character 'ﬀ', suggesting there may be broader Unicode handling issues in the implementation.

## Proposed Fix

The fix requires modifying the C implementation to use length-aware string operations instead of null-terminated C string functions. The implementation should:

1. Track string length explicitly rather than relying on null termination
2. Allocate output buffer based on input string length
3. Copy all characters including nulls to the output buffer
4. Apply case transformation only to cased characters, preserving non-cased characters like null

Without access to the C source, here's the conceptual approach:

```diff
// Pseudo-code for the C implementation fix
- char* result = process_string(input);  // Assumes null-terminated
- if (result[0] == '\0') return empty_string();
+ size_t len = get_string_length(input);  // Use actual length, not strlen
+ char* result = allocate_buffer(len);
+ for (size_t i = 0; i < len; i++) {
+     result[i] = toupper_unicode(input[i]);  // Process each character
+ }
+ return create_string_with_length(result, len);
```

A workaround at the Python level could wrap these functions to handle the edge case:

```python
def safe_upper(arr):
    """Wrapper that correctly handles null-only strings."""
    result = nps.upper(arr)
    # Check each element and restore null-only strings
    for i, (orig, res) in enumerate(zip(arr, result)):
        if orig and not res and all(c == '\x00' for c in orig):
            result[i] = orig.upper()  # Use Python's correct implementation
    return result
```