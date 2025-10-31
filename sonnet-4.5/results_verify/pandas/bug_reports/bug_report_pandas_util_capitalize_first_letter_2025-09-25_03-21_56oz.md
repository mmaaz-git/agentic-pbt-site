# Bug Report: pandas.util.capitalize_first_letter - Length Not Preserved with Unicode

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function violates the reasonable expectation that string length is preserved when only the first letter is capitalized. With certain Unicode characters (e.g., German 'ß'), the function increases string length because `'ß'.upper()` produces 'SS'.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.util import capitalize_first_letter

@given(st.text())
def test_capitalize_first_letter_length_preserved(s):
    result = capitalize_first_letter(s)
    assert len(result) == len(s), f"Input: {s!r}, Output: {result!r}"
```

**Failing input**: `'ß'`

## Reproducing the Bug

```python
from pandas.util import capitalize_first_letter

s = 'ß'
result = capitalize_first_letter(s)
print(f"Input: {s!r} (length {len(s)})")
print(f"Output: {result!r} (length {len(result)})")
assert len(result) == len(s), f"Expected length {len(s)}, got {len(result)}"
```

Output:
```
Input: 'ß' (length 1)
Output: 'SS' (length 2)
AssertionError: Expected length 1, got 2
```

## Why This Is A Bug

The function name `capitalize_first_letter` implies that it capitalizes only the first letter of the string, which would normally preserve string length. However, due to Unicode case mapping rules, certain characters like 'ß' expand to multiple characters ('SS') when uppercased, violating this expectation.

This is a contract violation because:
1. The function name suggests simple first-character capitalization
2. Users would reasonably expect `len(capitalize_first_letter(s)) == len(s)` to always hold
3. The function is exposed in the public pandas.util API without documentation warning about this behavior

## Fix

The most appropriate fix depends on the intended semantics:

**Option 1**: If length preservation is important, use character-level slicing that prevents expansion:

```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    if not s:
+        return s
+    first_char = s[0]
+    upper_first = first_char.upper()
+    # If uppercasing expands the character, use title case instead
+    if len(upper_first) != 1:
+        upper_first = first_char.title()
+    # If title case also expands, use original
+    if len(upper_first) != 1:
+        upper_first = first_char
+    return upper_first + s[1:]
```

**Option 2**: Add documentation clarifying the behavior:

```diff
 def capitalize_first_letter(s):
+    """
+    Uppercase the first character and leave the rest unchanged.
+
+    Note: Due to Unicode case mapping, the result may be longer than
+    the input (e.g., 'ß' becomes 'SS').
+
+    Parameters
+    ----------
+    s : str
+        Input string
+
+    Returns
+    -------
+    str
+        String with first character uppercased
+    """
     return s[:1].upper() + s[1:]
```

**Option 3**: Rename the function to better reflect its behavior (e.g., `uppercase_first_char`).

Given the single usage in `pandas/core/dtypes/dtypes.py` for dtype name comparison, the impact is minimal, but clarifying the behavior would prevent confusion for users of the public API.