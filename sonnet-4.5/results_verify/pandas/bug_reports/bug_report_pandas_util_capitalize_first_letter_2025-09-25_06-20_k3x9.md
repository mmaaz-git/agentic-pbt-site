# Bug Report: pandas.util.capitalize_first_letter Unicode Length Change

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function changes string length and fails to preserve the suffix for certain Unicode characters that have multi-character uppercase forms.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.util

@given(st.text(min_size=1))
def test_capitalize_first_letter_length_preservation(s):
    result = pandas.util.capitalize_first_letter(s)
    assert len(result) == len(s)
```

**Failing input**: `'ß'` (German sharp s)

## Reproducing the Bug

```python
import pandas.util

assert pandas.util.capitalize_first_letter('ß') == 'SS'
assert len('ß') == 1 and len('SS') == 2

assert pandas.util.capitalize_first_letter('ßeta') == 'SSeta'
assert 'ßeta'[1:] == 'eta' and 'SSeta'[1:] == 'Seta'

assert pandas.util.capitalize_first_letter('ﬁle') == 'FIle'
assert len('ﬁle') == 3 and len('FIle') == 4
```

## Why This Is A Bug

The function `capitalize_first_letter` is implemented as:

```python
def capitalize_first_letter(s):
    return s[:1].upper() + s[1:]
```

This implementation violates two properties implied by the function name:

1. **Length preservation**: Capitalizing the first letter should not change the string length, but it does for Unicode characters with multi-character uppercase forms (ß→SS, ﬁ→FI, ﬂ→FL, etc.)

2. **Suffix preservation**: The suffix `s[1:]` should remain unchanged, but it doesn't. For example, `'ßeta'` has suffix `'eta'`, but the result `'SSeta'` has suffix `'Seta'`.

While Python's built-in `str.upper()` correctly returns `'SS'` for `'ß'`, a function called `capitalize_first_letter` should either:
- Only capitalize the first character without changing its length
- Use a different approach that preserves the suffix
- Document this behavior clearly

## Fix

Use `str.capitalize()` instead, which handles this more gracefully (though it also lowercases the rest):

```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    if not s:
+        return s
+    # For single-char uppercase that expands to multiple chars,
+    # just return the original character capitalized via casefold
+    first_upper = s[0].upper()
+    if len(first_upper) > 1:
+        # Return original if uppercasing would change length
+        return s
+    return first_upper + s[1:]
```

Or if the current behavior is intentional, document it clearly:

```diff
 def capitalize_first_letter(s):
+    """
+    Capitalize the first letter of a string.
+
+    Note: For some Unicode characters (e.g., 'ß'), the uppercase form
+    may be multiple characters ('SS'), which will change the string length.
+    """
     return s[:1].upper() + s[1:]
```