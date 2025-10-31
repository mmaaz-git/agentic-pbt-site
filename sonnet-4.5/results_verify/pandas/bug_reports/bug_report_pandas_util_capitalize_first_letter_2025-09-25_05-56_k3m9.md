# Bug Report: pandas.util.capitalize_first_letter Length Change

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function changes the string length for certain Unicode characters (e.g., 'ß' becomes 'SS'), which violates the expectation that capitalizing only the first letter should preserve string length.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.util import capitalize_first_letter


@given(st.text())
def test_capitalize_first_letter_length_preserved(s):
    result = capitalize_first_letter(s)
    assert len(result) == len(s)
```

**Failing input**: `'ß'` (German sharp S)

## Reproducing the Bug

```python
from pandas.util import capitalize_first_letter

s = 'ß'
result = capitalize_first_letter(s)

print(f"Input: {s!r} (length {len(s)})")
print(f"Output: {result!r} (length {len(result)})")
```

Output:
```
Input: 'ß' (length 1)
Output: 'SS' (length 2)
```

## Why This Is A Bug

The function name `capitalize_first_letter` implies it should only affect the first letter/character without changing the string length. However, for certain Unicode characters like 'ß' (German sharp S) and 'ﬁ' (fi ligature), Python's `.upper()` method expands them into multiple characters ('SS' and 'FI' respectively).

While the function is currently only used internally with ASCII dtype names (like "period" -> "Period"), it's exposed in the public API (`pandas.util.capitalize_first_letter`), where users might call it with arbitrary Unicode strings and experience unexpected behavior.

## Fix

Use `.capitalize()` instead of `.upper()` for the first character, or use `.title()` for more sophisticated Unicode-aware capitalization:

```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    return s[:1].capitalize() + s[1:]
```

However, even `.capitalize()` has the same issue with certain Unicode characters. A more robust fix would be to handle the expansion explicitly:

```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    if not s:
+        return s
+    # Use str.capitalize() which is more appropriate for this use case
+    # It handles the first character and lowercases the rest, but since
+    # we only want to change the first character, we need a different approach
+    first = s[0]
+    rest = s[1:]
+    # For characters where .upper() changes length, keep original
+    if len(first.upper()) != len(first):
+        return s
+    return first.upper() + rest
```

Or simply document that the function may change string length for certain Unicode characters and is intended for ASCII strings only.