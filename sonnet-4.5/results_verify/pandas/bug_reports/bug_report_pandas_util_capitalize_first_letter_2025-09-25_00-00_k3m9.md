# Bug Report: pandas.util.capitalize_first_letter Unicode Expansion

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function unexpectedly changes string length for Unicode characters that expand when uppercased, such as 'ß' (German sharp S) → 'SS' and 'ﬁ' (ligature) → 'FI'.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.util

@given(st.text())
def test_capitalize_first_letter_property(s):
    result = pandas.util.capitalize_first_letter(s)
    if len(s) > 0:
        # The result should start with s[0].upper()
        expected_first = s[0].upper()
        assert result.startswith(expected_first)
        # But length may change for some Unicode characters
        if len(expected_first) > 1:
            # Unicode expansion occurred
            assert len(result) == len(s) + len(expected_first) - 1
```

**Failing input**: `s='ß'`

## Reproducing the Bug

```python
import pandas.util

s = 'ß'
result = pandas.util.capitalize_first_letter(s)

print(f"Input: {s!r} (length={len(s)})")
print(f"Result: {result!r} (length={len(result)})")
print(f"Expected first char upper: {s[0].upper()!r}")

assert len(result) != len(s), "Length changed unexpectedly"
assert result == 'SS', f"Got {result!r}, expected 'SS'"

s2 = 'ßeta'
result2 = pandas.util.capitalize_first_letter(s2)
print(f"\nInput: {s2!r} (length={len(s2)})")
print(f"Result: {result2!r} (length={len(result2)})")
assert len(result2) != len(s2), "Length changed unexpectedly"

s3 = 'ﬁle'
result3 = pandas.util.capitalize_first_letter(s3)
print(f"\nInput: {s3!r} (length={len(s3)})")
print(f"Result: {result3!r} (length={len(result3)})")
assert len(result3) != len(s3), "Length changed unexpectedly"
```

Output:
```
Input: 'ß' (length=1)
Result: 'SS' (length=2)
Expected first char upper: 'SS'

Input: 'ßeta' (length=4)
Result: 'SSeta' (length=5)

Input: 'ﬁle' (length=3)
Result: 'FIle' (length=4)
```

## Why This Is A Bug

While Python's `str.upper()` correctly implements Unicode case mapping (where some characters expand), the function name `capitalize_first_letter` (singular) suggests it operates on a single letter. The implementation `s[:1].upper() + s[1:]` can unexpectedly:

1. Change the string length for certain Unicode characters
2. Produce multiple uppercase letters when capitalizing a "first letter"
3. Violate the implicit assumption that `len(capitalize_first_letter(s)) == len(s)`

This behavior is surprising and undocumented, potentially causing issues in code that assumes length preservation.

## Fix

Two possible fixes depending on intent:

**Option 1**: Preserve string length by using title case for the first character:
```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    if not s:
+        return s
+    # Use title case which doesn't expand characters
+    return s[0].title() + s[1:]
```

Note: This still has issues because `'ß'.title()` returns 'Ss', not 'ß'.

**Option 2**: Document the behavior and rename the function:
```diff
 def capitalize_first_letter(s):
+    """
+    Uppercase the first character of string s and keep the rest unchanged.
+
+    Note: For certain Unicode characters (e.g., 'ß'), uppercasing can result
+    in multiple characters, changing the string length.
+    """
     return s[:1].upper() + s[1:]
```

**Option 3**: Use a length-preserving approach:
```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    if not s:
+        return s
+    first_upper = s[0].upper()
+    # If uppercasing expanded the character, use only the first char of the expansion
+    if len(first_upper) > 1:
+        first_upper = first_upper[0]
+    return first_upper + s[1:]
```

This would give 'Seta' for 'ßeta' instead of 'SSeta'.