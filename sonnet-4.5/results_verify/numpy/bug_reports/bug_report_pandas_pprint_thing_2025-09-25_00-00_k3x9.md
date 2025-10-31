# Bug Report: pandas.io.formats.printing.pprint_thing KeyError with Custom Escape Characters

**Target**: `pandas.io.formats.printing.pprint_thing`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pprint_thing` raises `KeyError` when `escape_chars` is a list containing characters other than `\t`, `\n`, or `\r`, despite the docstring indicating that any list of characters is acceptable.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.formats.printing as printing

@given(
    thing=st.text(),
    escape_chars=st.lists(
        st.sampled_from(['\t', '\n', '\r', 'a', 'b', 'c']),
        min_size=1,
        max_size=3
    )
)
@settings(max_examples=500)
def test_pprint_thing_escape_chars(thing, escape_chars):
    """Test that escape_chars parameter accepts any list of characters."""
    result = printing.pprint_thing(thing, escape_chars=escape_chars)
    assert isinstance(result, str)
```

**Failing input**: `thing='', escape_chars=['a']`

## Reproducing the Bug

```python
import pandas.io.formats.printing as printing

result = printing.pprint_thing("hello world", escape_chars=['a'])
```

**Error:**
```
KeyError: 'a'
```

**Root cause**: The internal `as_escaped_string` function only has default translations for `\t`, `\n`, and `\r`. When a list is passed as `escape_chars`, the code iterates over each character and tries to look it up in the `translate` dict:

```python
for c in escape_chars:
    result = result.replace(c, translate[c])  # KeyError if c not in translate
```

## Why This Is A Bug

1. **API contract violation**: The docstring states that `escape_chars` can be "list or dict", implying any list is acceptable, but only lists containing `\t`, `\n`, or `\r` work.

2. **Inconsistent behavior**: When `escape_chars` is a dict, any characters work. When it's a list, only specific characters work.

3. **Poor error message**: Users get a cryptic `KeyError` instead of a helpful error message about which characters are supported.

## Fix

The bug can be fixed by either rejecting unsupported characters with a clear error message, or by providing a default replacement. Here's a patch that provides a default replacement:

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -207,7 +207,10 @@ def pprint_thing(
             escape_chars = escape_chars or ()

         result = str(thing)
         for c in escape_chars:
-            result = result.replace(c, translate[c])
+            if c in translate:
+                result = result.replace(c, translate[c])
+            else:
+                result = result.replace(c, f"\\x{ord(c):02x}")
         return result

     if hasattr(thing, "__next__"):
```

Alternatively, raise a clear error:

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -207,7 +207,11 @@ def pprint_thing(
             escape_chars = escape_chars or ()

         result = str(thing)
         for c in escape_chars:
+            if c not in translate:
+                raise ValueError(
+                    f"Unsupported escape character {c!r}. "
+                    f"Supported characters when using a list: \\t, \\n, \\r. "
+                    f"Use a dict to specify custom replacements."
+                )
             result = result.replace(c, translate[c])
         return result
```