# Bug Report: Cython.Debugger is_valid_tag Missing XML Tag Validation

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_valid_tag()` function in `Cython.Debugger.DebugWriter` only validates the specific pattern of `.` followed by decimal digits (e.g., `.0`, `.123`) but does not validate general XML tag name rules. This causes `ValueError` exceptions when argument names start with digits or contain control characters, which are invalid in XML but pass `is_valid_tag()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

@given(st.text(alphabet=st.characters(min_codepoint=48, max_codepoint=57), min_size=1, max_size=1))
@settings(max_examples=10)
def test_is_valid_tag_should_reject_tags_starting_with_digit(digit_char):
    tag_name = EncodedString(digit_char + 'arg')
    result = is_valid_tag(tag_name)
    assert result == False, f'Expected False for tag starting with digit: {tag_name!r}, got {result}'
```

**Failing input**: `digit_char='0'` (creates tag name `'0arg'`)

## Reproducing the Bug

```python
import tempfile
from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = 'test_module'

    writer.start('Module')
    writer.start('Arguments')

    arg_name = EncodedString('0arg')

    print(f'is_valid_tag(arg_name): {is_valid_tag(arg_name)}')

    writer.start(arg_name)
    writer.end(arg_name)

    writer.end('Arguments')
    writer.serialize()
```

Output:
```
is_valid_tag(arg_name): True

ValueError: Invalid tag name '0arg'
```

## Why This Is A Bug

The `is_valid_tag()` function is used to filter out invalid XML tag names before passing them to the XML tree builder. In `ParseTreeTransforms.py` line 4414, argument names are used directly as XML tags:

```python
for arg in node.local_scope.arg_entries:
    self.tb.start(arg.name)
    self.tb.end(arg.name)
```

Since `is_valid_tag()` is called inside `start()` and `end()`, it should validate all XML tag name rules, not just the specific `.digit+` pattern. XML tag names:
1. Cannot start with a digit (e.g., `'0arg'`)
2. Cannot contain control characters (e.g., `'arg\x00name'`)
3. Cannot be purely numeric (e.g., `'0'`)

Currently, these invalid names pass `is_valid_tag()` but cause crashes when the XML tree builder processes them.

## Fix

```diff
--- a/Cython/Debugger/DebugWriter.py
+++ b/Cython/Debugger/DebugWriter.py
@@ -1,5 +1,6 @@
 import os
 import sys
 import errno
+import re

 try:
     from lxml import etree
@@ -15,14 +16,28 @@ from ..Compiler.StringEncoding import EncodedString

 def is_valid_tag(name):
     """
     Names like '.0' are used internally for arguments
     to functions creating generator expressions,
     however they are not identifiers.

     See https://github.com/cython/cython/issues/5552
+
+    Additionally, XML tag names must follow XML naming rules:
+    - Cannot start with a digit
+    - Cannot contain control characters
+    - Must be valid XML Name tokens
     """
     if isinstance(name, EncodedString):
         if name.startswith(".") and name[1:].isdecimal():
             return False
+
+        # Check XML tag name validity
+        # XML names cannot start with digits
+        if name and name[0].isdigit():
+            return False
+
+        # Check for control characters (0x00-0x1F, except tab, newline, carriage return)
+        if any(ord(c) < 32 and c not in '\t\n\r' for c in name):
+            return False
+
     return True
```