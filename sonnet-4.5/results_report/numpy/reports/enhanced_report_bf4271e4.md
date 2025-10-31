# Bug Report: numpy.char.upper/lower Silent Data Loss with Unicode Case Expansion

**Target**: `numpy.char.upper`, `numpy.char.lower`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper` and `numpy.char.lower` silently truncate Unicode characters that expand during case transformation based on the array's dtype, causing data loss without warning.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings


safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=('Cs',),
        blacklist_characters=' \t\n\r\x00\x0b\x0c'
    ),
    min_size=1
)


@given(safe_text)
@settings(max_examples=1000)
def test_upper_lower_roundtrip(s):
    arr = char.array([s])
    result = char.lower(char.upper(arr))
    expected = s.upper().lower()
    assert result[0] == expected


if __name__ == "__main__":
    test_upper_lower_roundtrip()
```

<details>

<summary>
**Failing input**: `'ß'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 24, in <module>
    test_upper_lower_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 15, in test_upper_lower_roundtrip
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 20, in test_upper_lower_roundtrip
    assert result[0] == expected
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_upper_lower_roundtrip(
    s='ß',
)
```
</details>

## Reproducing the Bug

```python
import numpy.char as char

s = 'ῂ'
arr = char.array([s])
upper = char.upper(arr)
lower_upper = char.lower(upper)

python_upper = s.upper()
python_lower_upper = python_upper.lower()

print(f"Input: {repr(s)}")
print(f"Python:  {repr(s)} -> upper: {repr(python_upper)} -> lower: {repr(python_lower_upper)}")
print(f"NumPy:   {repr(s)} -> upper: {repr(upper[0])} -> lower: {repr(lower_upper[0])}")
print(f"NumPy result: {repr(lower_upper[0])}")
print(f"Expected:     {repr(python_lower_upper)}")
print()
print(f"Python produces {len(python_lower_upper)} characters: {repr(python_lower_upper)}")
print(f"NumPy produces {len(lower_upper[0])} character: {repr(lower_upper[0])}")
print()
print("Data Loss: The iota character is completely lost in NumPy's transformation")
```

<details>

<summary>
Greek character with iota subscript loses data
</summary>
```
Input: 'ῂ'
Python:  'ῂ' -> upper: 'ῊΙ' -> lower: 'ὴι'
NumPy:   'ῂ' -> upper: np.str_('Ὴ') -> lower: np.str_('ὴ')
NumPy result: np.str_('ὴ')
Expected:     'ὴι'

Python produces 2 characters: 'ὴι'
NumPy produces 1 character: np.str_('ὴ')

Data Loss: The iota character is completely lost in NumPy's transformation
```
</details>

## Why This Is A Bug

This behavior violates the documented contract that `numpy.char.upper()` and `numpy.char.lower()` "call `str.upper`/`str.lower` element-wise". The actual behavior silently truncates results based on the array's dtype, causing data loss without warning.

The root cause is that `numpy.char.array()` automatically chooses dtype `U1` (single Unicode character) for single-character inputs. When Unicode case transformations expand the string (e.g., German 'ß' → 'SS', Greek 'ῂ' → 'ῊΙ'), the result is truncated to fit the dtype.

This violates:
1. **Documentation**: States it calls Python's string methods element-wise, implying identical behavior
2. **Unicode Standards**: Proper Unicode case transformations per Unicode Technical Report #21
3. **Data Integrity**: Silent data loss without warning is dangerous
4. **User Expectations**: NumPy should match Python's string behavior or warn about differences

## Relevant Context

The bug affects common real-world scenarios:
- **German text**: The eszett 'ß' uppercases to 'SS', but NumPy truncates to 'S'
- **Greek text**: Characters with iota subscripts (ῂ, ῃ, ῴ, etc.) expand when uppercased
- **Turkish**: The dotted capital İ has special case rules

The char module implementation uses `_vec_string` from `numpy._core.multiarray` (C code) which respects dtype constraints but doesn't handle expansion. The issue is documented in:
- `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/_core/strings.py:1102-1173`
- The char module is marked as "legacy" and not recommended for new development

Workaround: Users can specify a larger dtype to avoid truncation:
```python
# This preserves the full result
arr = np.array(['ß'], dtype='U10')
char.upper(arr)  # Returns 'SS' correctly
```

## Proposed Fix

Since the char module uses fixed-size dtypes, a complete fix requires significant changes. Here's a high-level approach:

1. **Immediate fix**: Add a warning when case transformations would cause truncation
2. **Better fix**: Automatically resize the output array dtype when needed
3. **Documentation fix**: Clearly document dtype limitations and truncation behavior

A minimal warning patch would be:

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1132,7 +1132,15 @@ def upper(a):

     """
     a_arr = np.asarray(a)
-    return _vec_string(a_arr, a_arr.dtype, 'upper')
+    result = _vec_string(a_arr, a_arr.dtype, 'upper')
+
+    # Check for truncation by comparing with Python's str.upper
+    if a_arr.dtype.kind == 'U':
+        for orig, res in zip(a_arr.flat, result.flat):
+            if len(str(orig).upper()) > len(str(res)):
+                import warnings
+                warnings.warn(f"Unicode case transformation truncated from '{str(orig).upper()}' to '{res}' due to dtype {a_arr.dtype}", UserWarning, stacklevel=2)
+                break
+    return result


 @set_module("numpy.strings")
```

However, the ideal fix would be to use dynamic string allocation or at least pre-calculate the required dtype size before transformation.