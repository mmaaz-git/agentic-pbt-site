# Bug Report: django.utils.text.wrap Silently Removes Line Separator Characters

**Target**: `django.utils.text.wrap`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `wrap()` function in `django.utils.text` silently removes non-whitespace line separator characters (U+001C File Separator, U+001D Group Separator, U+001E Record Separator) from input text, violating its documented behavior of preserving content.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from django.utils.text import wrap


@settings(max_examples=500)
@given(st.text(), st.integers(min_value=10, max_value=200))
@example('\x1e', 10)
@example('Data\x1eRecord', 100)
def test_wrap_preserves_content(text, width):
    wrapped = wrap(text, width)
    assert wrapped.replace('\n', '') == text.replace('\n', ''), \
        f"Content was not preserved: {repr(text)} -> {repr(wrapped)}"
```

**Failing input**: `'\x1e'` (Record Separator, U+001E)

## Reproducing the Bug

```python
from django.utils.text import wrap

text = '\x1e'
result = wrap(text, 100)

print(f"Input:  {repr(text)}")
print(f"Output: {repr(result)}")
assert text == result


text2 = 'Record1\x1eRecord2'
result2 = wrap(text2, 100)
print(f"Input:  {repr(text2)}")
print(f"Output: {repr(result2)}")
```

Output:
```
Input:  '\x1e'
Output: ''
AssertionError: Content changed

Input:  'Record1\x1eRecord2'
Output: 'Record1\nRecord2'
```

## Why This Is A Bug

1. **Documented behavior violation**: The `wrap()` docstring states: "Preserve all white space except added line breaks consume the space on which they break the line." However, U+001C, U+001D, and U+001E are **NOT whitespace** (they fail `str.isspace()`), yet they are being removed.

2. **Data corruption**: These are legitimate ASCII control characters used in data formats. Legacy systems and some data formats use these separators to structure data. Silently removing them corrupts the data.

3. **Silent failure**: The function provides no indication that content was removed, making this bug difficult to detect.

4. **Inconsistent with line breaks**: Regular line breaks (`\n`) are preserved in the output (converted to `\n`), but other line separators are silently deleted.

## Root Cause

The bug occurs because `wrap()` uses `text.splitlines()` which recognizes U+001C/1D/1E as line boundaries. When these characters appear alone or at boundaries, the resulting split produces empty strings that get filtered out. The function at line 63-64 attempts to restore lines with only whitespace, but these characters are not whitespace, so they're not restored.

```python
# From django/utils/text.py lines 60-70
for line in text.splitlines():
    wrapped = wrapper.wrap(line)
    if not wrapped:
        # If `line` contains only whitespaces that are dropped, restore it.
        result.append(line)  # This restores empty lines, not the separator!
    else:
        result.extend(wrapped)
```

When `'\x1e'.splitlines()` returns `['', '']`, both are empty strings, not the original separator character.

## Fix

The function should preserve line separator characters by either:
1. Not using `splitlines()` for splitting (split only on \n/\r\n)
2. Preserving the separator characters when joining

Suggested patch:

```diff
--- a/django/utils/text.py
+++ b/django/utils/text.py
@@ -57,7 +57,14 @@ def wrap(text, width):
         replace_whitespace=False,
     )
     result = []
-    for line in text.splitlines():
+    # Use regex to split on common newlines only, not all Unicode line separators
+    # This preserves U+001C/1D/1E and other separators as data
+    import re
+    lines = re.split(r'(\r\n|\r|\n)', text)
+    for i, line in enumerate(lines):
+        # Preserve the newline separators themselves
+        if line in ('\r\n', '\r', '\n'):
+            result.append(line.replace('\r\n', '\n').replace('\r', '\n'))
+            continue
         wrapped = wrapper.wrap(line)
         if not wrapped:
             # If `line` contains only whitespaces that are dropped, restore it.
```

This preserves legacy ASCII separators as data while maintaining the documented behavior.