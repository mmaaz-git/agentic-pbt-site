# Bug Report: numpy.char Case Conversion Functions Silently Truncate Unicode Expansions

**Target**: `numpy.char.upper`, `numpy.char.lower`, `numpy.char.swapcase`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy's character case conversion functions silently truncate results when Unicode case conversion produces strings longer than the input array's dtype width, violating documented behavior and mathematical properties.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st

@given(st.just('ß'))
def test_swapcase_involution(text):
    """Test that swapcase is an involution (applying it twice returns original)"""
    arr = np.array([text])
    result = numpy.char.swapcase(numpy.char.swapcase(arr))
    assert result[0] == text, f"swapcase(swapcase('{text}')) = {repr(result[0])}, expected '{text}'"

@given(st.just('ß'))
def test_upper_lower_idempotence(text):
    """Test that lower(upper(x)) = lower(x) for idempotence property"""
    arr = np.array([text])
    result1 = numpy.char.lower(numpy.char.upper(arr))
    result2 = numpy.char.lower(arr)
    assert result1[0] == result2[0], f"lower(upper('{text}')) = {repr(result1[0])}, lower('{text}') = {repr(result2[0])}"

if __name__ == "__main__":
    print("Running test_swapcase_involution with input 'ß'...")
    try:
        test_swapcase_involution()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nRunning test_upper_lower_idempotence with input 'ß'...")
    try:
        test_upper_lower_idempotence()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: `'ß'`
</summary>
```
Running test_swapcase_involution with input 'ß'...
Test failed: swapcase(swapcase('ß')) = np.str_('s'), expected 'ß'

Running test_upper_lower_idempotence with input 'ß'...
Test failed: lower(upper('ß')) = np.str_('s'), lower('ß') = np.str_('ß')
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char

# Test case 1: Upper case conversion of 'ß'
arr = np.array(['ß'])
result = numpy.char.upper(arr)
print(f"numpy.char.upper(['ß']) returns: {repr(result[0])}")
print(f"Expected: 'SS', Got: {repr(result[0])}")
print()

# Test case 2: Swapcase involution property
swapped_once = numpy.char.swapcase(arr)
print(f"numpy.char.swapcase(['ß']) returns: {repr(swapped_once[0])}")
swapped_twice = numpy.char.swapcase(swapped_once)
print(f"numpy.char.swapcase(numpy.char.swapcase(['ß'])) returns: {repr(swapped_twice[0])}")
print(f"Expected: 'ß', Got: {repr(swapped_twice[0])}")
print()

# Test case 3: Upper/Lower idempotence property
upper = numpy.char.upper(arr)
upper_lower = numpy.char.lower(upper)
print(f"numpy.char.lower(numpy.char.upper(['ß'])) returns: {repr(upper_lower[0])}")
print(f"Expected: 'ß', Got: {repr(upper_lower[0])}")
print()

# Show Python's behavior for comparison
print("Python's str methods:")
print(f"'ß'.upper() = {repr('ß'.upper())}")
print(f"'ß'.swapcase() = {repr('ß'.swapcase())}")
print(f"'ß'.swapcase().swapcase() = {repr('ß'.swapcase().swapcase())}")
print(f"'ß'.upper().lower() = {repr('ß'.upper().lower())}")
```

<details>

<summary>
Output showing truncation to single character
</summary>
```
numpy.char.upper(['ß']) returns: np.str_('S')
Expected: 'SS', Got: np.str_('S')

numpy.char.swapcase(['ß']) returns: np.str_('S')
numpy.char.swapcase(numpy.char.swapcase(['ß'])) returns: np.str_('s')
Expected: 'ß', Got: np.str_('s')

numpy.char.lower(numpy.char.upper(['ß'])) returns: np.str_('s')
Expected: 'ß', Got: np.str_('s')

Python's str methods:
'ß'.upper() = 'SS'
'ß'.swapcase() = 'SS'
'ß'.swapcase().swapcase() = 'ss'
'ß'.upper().lower() = 'ss'
```
</details>

## Why This Is A Bug

This behavior violates the documented contract and causes silent data corruption:

1. **Documentation Contradiction**: The numpy.char.upper() documentation explicitly states it "Calls str.upper element-wise." However, Python's `str.upper('ß')` returns `'SS'` while NumPy returns `'S'` when using auto-inferred dtype.

2. **Silent Data Loss**: The function truncates the result without warning or error. When NumPy auto-infers dtype as `<U1` for the single character 'ß', the uppercase result 'SS' gets silently truncated to 'S'.

3. **Mathematical Property Violations**:
   - **Involution property broken**: `swapcase(swapcase(x))` should equal `x`, but `swapcase(swapcase('ß'))` returns `'s'` instead of `'ß'`
   - **Idempotence property broken**: `lower(upper(x))` should equal `lower(x)`, but returns `'s'` instead of `'ß'`

4. **Unicode Standard Violation**: The Unicode standard defines the correct uppercase mapping for 'ß' (U+00DF LATIN SMALL LETTER SHARP S) as 'SS'. This affects real-world text processing for German and other languages.

## Relevant Context

The bug occurs because NumPy uses fixed-width string dtypes. When creating `np.array(['ß'])`, NumPy infers dtype `<U1` (Unicode string of length 1). When upper() produces 'SS' (length 2), it gets truncated to fit the original dtype width.

Testing confirms the dtype-dependent behavior:
- `np.array(['ß'])` with auto-inferred dtype `<U1`: upper() returns `'S'` (truncated)
- `np.array(['ß'], dtype='U2')` with explicit dtype `<U2`: upper() returns `'SS'` (correct)

Affected Unicode characters include:
- `'ß'` (German Eszett) → should become `'SS'`
- `'ﬃ'` (ffi ligature) → should become `'FFI'`
- `'ﬁ'` (fi ligature) → should become `'FI'`
- `'ﬆ'` (st ligature) → should become `'ST'`

Documentation links:
- numpy.char.upper: https://numpy.org/doc/stable/reference/generated/numpy.char.upper.html
- Source code: numpy/_core/strings.py:1102-1135 (upper function implementation)

## Proposed Fix

The issue requires detecting when case conversion would expand the string beyond the current dtype width. Here's a high-level approach:

1. Pre-calculate the maximum length after case conversion for all elements
2. Allocate output array with appropriate dtype width
3. Return results without truncation

Since this would be a significant change to NumPy's string handling, a more practical immediate fix would be to add runtime warnings when truncation occurs:

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1133,7 +1133,15 @@ def upper(a):

     """
     a_arr = np.asarray(a)
-    return _vec_string(a_arr, a_arr.dtype, 'upper')
+    result = _vec_string(a_arr, a_arr.dtype, 'upper')
+
+    # Check if any truncation occurred
+    if a_arr.dtype.kind == 'U':
+        for i, elem in enumerate(a_arr.flat):
+            expected = str(elem).upper()
+            if len(expected) > a_arr.dtype.itemsize // 4:  # Unicode uses 4 bytes per char
+                import warnings
+                warnings.warn(f"String truncation occurred: '{elem}' -> '{expected}' truncated to fit dtype {a_arr.dtype}", RuntimeWarning)
+    return result
```

Similar changes would be needed for `lower()` and `swapcase()` functions.