# Bug Report: Cython.Debugger.DebugWriter.is_valid_tag Incomplete XML Validation

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_valid_tag()` function fails to properly validate XML tag names, allowing invalid tags to reach lxml's TreeBuilder which then crashes. The function only checks the `.N` pattern for `EncodedString` instances, missing multiple categories of invalid XML tags including regular strings with `.N` pattern, digit-starting tags, and control characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import CythonDebugWriter
import tempfile


@given(st.integers(min_value=0, max_value=100))
def test_invalid_tag_names_should_not_crash(n):
    tag_name = f'.{n}'
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        writer.module_name = "test_module"
        writer.start(tag_name)
        writer.end(tag_name)
```

**Failing input**: `n=0` (produces tag name `.0`)

## Reproducing the Bug

```python
import tempfile
from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag

print("Case 1: '.0' as regular string")
print(f"is_valid_tag('.0') = {is_valid_tag('.0')}")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    try:
        writer.start('.0')
    except ValueError as e:
        print(f"Crashes with: {e}")

print("\nCase 2: Digit-starting tag '0'")
print(f"is_valid_tag('0') = {is_valid_tag('0')}")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    try:
        writer.start('0')
    except ValueError as e:
        print(f"Crashes with: {e}")

print("\nCase 3: Control character")
print(f"is_valid_tag('\\x1f') = {is_valid_tag(chr(0x1f))}")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    try:
        writer.start('\x1f')
    except ValueError as e:
        print(f"Crashes with: {e}")
```

## Why This Is A Bug

The `is_valid_tag()` function serves as a validation guard in `CythonDebugWriter` methods (`start()`, `end()`, `add_entry()`) to prevent invalid XML tag names from reaching lxml's `TreeBuilder`. However, the validation is incomplete and allows multiple categories of invalid XML tags:

1. **`.N` pattern for regular strings**: The docstring mentions filtering "names like '.0'" but only does so for `EncodedString` instances, not regular strings
2. **Digit-starting tags**: XML tag names cannot start with digits, but `is_valid_tag('0')` returns `True`
3. **Control characters**: XML doesn't allow control characters in tag names, but `is_valid_tag('\x1f')` returns `True`

When these invalid tags reach lxml, they cause `ValueError` exceptions, crashing the debug info generation process. This affects the Cython compilation workflow when debug information is enabled.

## Fix

```diff
--- a/Cython/Debugger/DebugWriter.py
+++ b/Cython/Debugger/DebugWriter.py
@@ -18,9 +18,20 @@ def is_valid_tag(name):

     See https://github.com/cython/cython/issues/5552
     """
-    if isinstance(name, EncodedString):
-        if name.startswith(".") and name[1:].isdecimal():
-            return False
+    if not isinstance(name, (str, EncodedString)):
+        return False
+
+    if len(name) == 0:
+        return False
+
+    if len(name) > 1 and name.startswith(".") and name[1:].isdecimal():
+        return False
+
+    if name[0].isdigit():
+        return False
+
+    if any(ord(c) < 32 for c in name):
+        return False
+
     return True
```