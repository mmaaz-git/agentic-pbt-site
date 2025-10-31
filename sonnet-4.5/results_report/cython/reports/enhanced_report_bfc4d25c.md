# Bug Report: Cython.Debugger.DebugWriter.CythonDebugWriter.add_entry Crashes on Control Characters in Attribute Values

**Target**: `Cython.Debugger.DebugWriter.CythonDebugWriter.add_entry`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `add_entry` method in `CythonDebugWriter` crashes when attribute values contain control characters (ASCII < 32), failing to validate XML compatibility before passing values to the underlying XML TreeBuilder.

## Property-Based Test

```python
import tempfile
import shutil
from hypothesis import given, strategies as st, settings
from Cython.Debugger.DebugWriter import CythonDebugWriter

# Define valid XML tag name strategies
valid_xml_start = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
valid_xml_chars = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
valid_xml_tag = st.builds(
    lambda start, rest: start + rest,
    valid_xml_start,
    st.text(alphabet=valid_xml_chars, max_size=20)
)

@given(valid_xml_tag, st.dictionaries(valid_xml_tag, st.text()))
@settings(max_examples=100)
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
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    # Run the test
    test_add_entry_with_attrs()
```

<details>

<summary>
**Failing input**: `tag_name='a', attrs={'a': '\x1f'}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 31, in <module>
    test_add_entry_with_attrs()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 16, in test_add_entry_with_attrs
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 22, in test_add_entry_with_attrs
    writer.add_entry(tag_name, **attrs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Debugger/DebugWriter.py", line 58, in add_entry
    self.tb.start(name, attrs)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "src/lxml/saxparser.pxi", line 849, in lxml.etree.TreeBuilder.start
  File "src/lxml/saxparser.pxi", line 777, in lxml.etree.TreeBuilder._handleSaxStart
  File "src/lxml/apihelpers.pxi", line 206, in lxml.etree._makeSubElement
  File "src/lxml/apihelpers.pxi", line 201, in lxml.etree._makeSubElement
  File "src/lxml/apihelpers.pxi", line 325, in lxml.etree._initNodeAttributes
  File "src/lxml/apihelpers.pxi", line 336, in lxml.etree._addAttributeToNode
  File "src/lxml/apihelpers.pxi", line 1530, in lxml.etree._utf8
ValueError: All strings must be XML compatible: Unicode or ASCII, no NULL bytes or control characters
Falsifying example: test_add_entry_with_attrs(
    tag_name='a',
    attrs={'a': '\x1f'},
)
```
</details>

## Reproducing the Bug

```python
import tempfile
import shutil
from Cython.Debugger.DebugWriter import CythonDebugWriter

tmpdir = tempfile.mkdtemp()

try:
    writer = CythonDebugWriter(tmpdir)
    writer.tb.start('Module', {})

    # Try to add an entry with a control character in an attribute value
    writer.add_entry('a', a='\x1f')

    writer.tb.end('Module')
    writer.tb.end('cython_debug')
    writer.tb.close()
    print("No crash - unexpected")
except ValueError as e:
    print(f"Crashed with ValueError: {e}")
except Exception as e:
    print(f"Crashed with {type(e).__name__}: {e}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

<details>

<summary>
Crashed with ValueError from lxml validation
</summary>
```
Crashed with ValueError: All strings must be XML compatible: Unicode or ASCII, no NULL bytes or control characters
```
</details>

## Why This Is A Bug

The `add_entry` method accepts arbitrary keyword arguments (`**attrs`) without documenting or enforcing XML compatibility requirements. The method signature implies it can handle any string values, but it crashes when attribute values contain control characters (ASCII codes < 32). The error occurs deep in lxml's XML validation rather than at the API boundary, providing no graceful error handling or clear documentation about input restrictions. Since this is used internally by Cython to generate debug information during compilation, the compiler could encounter strings with control characters from source code literals, generated variable names, or file paths, causing debug information generation to fail unexpectedly.

## Relevant Context

The `CythonDebugWriter` class is used by Cython's compiler to generate XML debug information files that enable debugging of Cython-compiled code with tools like cygdb. The class documentation states it "writes debug information to cython_debug/cython_debug_info_<modulename>" but doesn't mention XML character restrictions.

Looking at the implementation in `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Debugger/DebugWriter.py`, the `add_entry` method (lines 56-59) validates tag names using `is_valid_tag()` but passes attribute values directly to the XML TreeBuilder without validation. The `is_valid_tag` function only checks for specific internal naming patterns, not XML compatibility.

XML 1.0 specification prohibits control characters in attribute values, and lxml strictly enforces this. The bug manifests when lxml's `_utf8` function validates the string during XML tree construction.

## Proposed Fix

```diff
--- a/Cython/Debugger/DebugWriter.py
+++ b/Cython/Debugger/DebugWriter.py
@@ -27,6 +27,18 @@ def is_valid_tag(name):
     return True


+def sanitize_xml_attrs(attrs):
+    """
+    Filter out or sanitize attribute values that are not XML-compatible.
+    Control characters (ASCII < 32) are not allowed in XML attribute values.
+    """
+    sanitized = {}
+    for key, value in attrs.items():
+        if not isinstance(value, str):
+            value = str(value)
+        # Filter out control characters
+        sanitized[key] = ''.join(c for c in value if ord(c) >= 32 or c in '\t\n\r')
+    return sanitized
+
 class CythonDebugWriter:
     """
     Class to output debugging information for cygdb
@@ -55,7 +67,8 @@ class CythonDebugWriter:

     def add_entry(self, name, **attrs):
         if is_valid_tag(name):
-            self.tb.start(name, attrs)
+            safe_attrs = sanitize_xml_attrs(attrs)
+            self.tb.start(name, safe_attrs)
             self.tb.end(name)

     def serialize(self):
```