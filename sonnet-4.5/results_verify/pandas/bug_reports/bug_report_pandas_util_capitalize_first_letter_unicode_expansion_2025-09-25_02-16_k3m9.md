# Bug Report: pandas.util.capitalize_first_letter Unicode Character Expansion

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function violates length preservation for certain Unicode characters that expand when uppercased, such as German ß (sharp s) and ligatures like ﬁ and ﬂ.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.util


@given(st.text())
def test_capitalize_preserves_length(s):
    result = pandas.util.capitalize_first_letter(s)
    assert len(result) == len(s)
```

**Failing inputs**:
- `'ß'` → `'SS'` (length 1 → 2)
- `'ﬁle'` → `'FIle'` (length 3 → 4)
- `'ﬂow'` → `'FLow'` (length 3 → 4)

## Reproducing the Bug

```python
import pandas.util

result = pandas.util.capitalize_first_letter('ßeta')
print(f"Input:  'ßeta' (length 4)")
print(f"Output: '{result}' (length {len(result)})")

result2 = pandas.util.capitalize_first_letter('ﬁle')
print(f"Input:  'ﬁle' (length 3)")
print(f"Output: '{result2}' (length {len(result2)})")
```

Output:
```
Input:  'ßeta' (length 4)
Output: 'SSeta' (length 5)
Input:  'ﬁle' (length 3)
Output: 'FIle' (length 4)
```

## Why This Is A Bug

The function name `capitalize_first_letter` implies it operates on a single letter/character, preserving the overall structure of the string. The implementation `s[:1].upper() + s[1:]` causes certain Unicode characters to expand:

1. German ß (U+00DF) uppercases to 'SS' (two characters)
2. Ligature ﬁ (U+FB01) uppercases to 'FI' (two characters)
3. Ligature ﬂ (U+FB02) uppercases to 'FL' (two characters)

This violates the reasonable expectation that capitalizing a string shouldn't change its length, which can cause issues in code that assumes length preservation (e.g., string formatting, padding, truncation).

## Fix

Use `str.capitalize()` which handles these cases more intuitively, or explicitly document the expansion behavior:

```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    # Handle Unicode characters that expand on uppercase
+    if not s:
+        return s
+    # Use capitalize() which lowercases the rest but handles expansion correctly
+    # Or use casefold/titlecase for more sophisticated handling
+    return s[0].upper() + s[1:] if len(s[0].upper()) == 1 else s.capitalize()
```

Alternatively, use Python's built-in `str.capitalize()` which handles this more gracefully:

```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    # Note: capitalize() lowercases all characters after the first
+    # If preserving case of remaining chars is needed, keep current impl
+    # but document the expansion behavior for certain Unicode chars
+    return s.capitalize()
```

Note: `str.capitalize()` also expands these characters but lowercases the rest, which may not be the desired behavior. If the intent is to preserve the case of all characters except the first, the expansion behavior should at least be documented.