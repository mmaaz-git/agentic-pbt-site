# Bug Report: Cython.Debugger.DebugWriter.add_entry Missing Attribute Value Validation

**Target**: `Cython.Debugger.DebugWriter.CythonDebugWriter.add_entry`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `add_entry` method in `CythonDebugWriter` accepts arbitrary attribute values without validating them for XML compatibility, causing crashes when attribute values contain control characters.

## Property-Based Test

```python
import tempfile
from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import CythonDebugWriter

valid_xml_start = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
valid_xml_chars = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
valid_xml_tag = st.builds(
    lambda start, rest: start + rest,
    valid_xml_start,
    st.text(alphabet=valid_xml_chars, max_size=20)
)

@given(valid_xml_tag, st.dictionaries(valid_xml_tag, st.text()))
def test_add_entry_with_attrs(tag_name, attrs):
    tmpdir = tempfile.mkdtemp()
    try:
        writer = CythonDebugWriter(tmpdir)
        writer.tb.start('Module', {})
        writer.add_entry(tag_name, **attrs)
        writer.tb.end('Module')
        writer.tb.end('cython_debug')
        writer.tb.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing input**: `tag_name='a', attrs={'a': '\x1f'}`

## Reproducing the Bug

```python
import tempfile
from Cython.Debugger.DebugWriter import CythonDebugWriter

tmpdir = tempfile.mkdtemp()

writer = CythonDebugWriter(tmpdir)
writer.tb.start('Module', {})

try:
    writer.add_entry('valid_tag', attr_with_control_char='\x1f')
    print("No crash - unexpected")
except ValueError as e:
    print(f"Bug: Crashed with: {e}")

import shutil
shutil.rmtree(tmpdir)
```

## Why This Is A Bug

The `add_entry()` method is meant to safely add XML elements with attributes. However, it doesn't validate attribute values for XML compatibility. XML attribute values cannot contain control characters (characters with ASCII codes < 32), but the method accepts any string value, leading to crashes when serializing the XML tree.

This is a crash bug because valid usage (adding debug information with arbitrary string values) can cause the entire debug writer to fail.

## Fix

```diff
--- a/Cython/Debugger/DebugWriter.py
+++ b/Cython/Debugger/DebugWriter.py
@@ -16,6 +16,13 @@ from ..Compiler.StringEncoding import EncodedString
 def is_valid_tag(name):
     ...

+def is_valid_attr_value(value):
+    """Check if a string is valid for use as an XML attribute value"""
+    if not isinstance(value, (str, EncodedString)):
+        value = str(value)
+    return all(ord(c) >= 32 for c in value)
+
+
 class CythonDebugWriter:
     ...

     def add_entry(self, name, **attrs):
         if is_valid_tag(name):
+            # Filter out attributes with invalid values
+            safe_attrs = {k: v for k, v in attrs.items() if is_valid_attr_value(v)}
-            self.tb.start(name, attrs)
+            self.tb.start(name, safe_attrs)
             self.tb.end(name)
```