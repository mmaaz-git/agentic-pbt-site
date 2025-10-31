# Bug Report: Cython.Debugger.DebugWriter.is_valid_tag - Incomplete XML Tag Validation

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_valid_tag` function claims to validate tag names but only checks for a specific pattern (EncodedStrings starting with "." followed by decimals). It fails to detect many invalid XML tag names, causing crashes when these tags are used with `CythonDebugWriter.start()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Debugger.DebugWriter import is_valid_tag, CythonDebugWriter
import tempfile


@given(st.text())
@settings(max_examples=1000)
def test_is_valid_tag_matches_xml_validity(tag_name):
    is_valid_result = is_valid_tag(tag_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        try:
            writer.start(tag_name)
            xml_accepts = True
        except ValueError:
            xml_accepts = False

    if is_valid_result and not xml_accepts:
        raise AssertionError(
            f"is_valid_tag({tag_name!r}) returned True, "
            f"but XML rejected it as invalid tag"
        )
```

**Failing input**: `'0'` (and many others: `'123'`, `'-invalid'`, `'with space'`, `'with:colon'`, `'.'`, etc.)

## Reproducing the Bug

```python
import sys
import tempfile
sys.path.insert(0, '/path/to/site-packages')

from Cython.Debugger.DebugWriter import is_valid_tag, CythonDebugWriter

tag = "0"
print(f"is_valid_tag('{tag}') = {is_valid_tag(tag)}")

with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.start(tag)
```

Output:
```
is_valid_tag('0') = True
ValueError: Invalid tag name '0'
```

## Why This Is A Bug

The function is named `is_valid_tag` and is used to filter invalid tag names before passing them to XML TreeBuilder. However, it only checks for one specific invalid pattern (EncodedStrings like ".0", ".1" etc.) while missing many other invalid XML tag names.

XML tag names must:
- Start with a letter or underscore (not digits or hyphens)
- Not contain spaces
- Not contain colons in certain contexts

Tags like "0", "123", "-invalid", "with space", "with:colon", "." all pass `is_valid_tag` but crash with `ValueError: Invalid tag name` when used.

This violates the contract implied by the function name and its usage in `start()`, `end()`, and `add_entry()` methods which call `is_valid_tag` to filter invalid tags.

## Fix

```diff
--- a/Cython/Debugger/DebugWriter.py
+++ b/Cython/Debugger/DebugWriter.py
@@ -15,6 +15,7 @@ from ..Compiler.StringEncoding import EncodedString

 def is_valid_tag(name):
     """
     Names like '.0' are used internally for arguments
     to functions creating generator expressions,
     however they are not identifiers.
+
+    Additionally, XML tag names have specific requirements:
+    - Must start with a letter or underscore
+    - Cannot contain spaces or certain special characters

     See https://github.com/cython/cython/issues/5552
     """
+    if not name:
+        return False
+
     if isinstance(name, EncodedString):
         if name.startswith(".") and name[1:].isdecimal():
             return False
+
+    name_str = str(name)
+    if not name_str:
+        return False
+
+    first_char = name_str[0]
+    if not (first_char.isalpha() or first_char == '_'):
+        return False
+
+    if ' ' in name_str or ':' in name_str:
+        return False
+
     return True
```