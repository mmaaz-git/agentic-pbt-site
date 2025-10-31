# Bug Report: Cython.Debugger.DebugWriter.is_valid_tag - Incomplete XML Validation

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_valid_tag()` function only validates against the specific pattern ".DECIMAL" (e.g., ".0", ".123") but fails to validate that tag names are valid XML identifiers, causing `CythonDebugWriter` methods to crash with `ValueError` when processing many common invalid tag names.

## Property-Based Test

```python
from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag
from hypothesis import given, strategies as st, settings
import tempfile
import shutil


@given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=20))
@settings(max_examples=500)
def test_start_end_symmetry(tag_names):
    tmpdir = tempfile.mkdtemp()
    try:
        writer = CythonDebugWriter(tmpdir)
        writer.module_name = "test_module"

        for tag in tag_names:
            if is_valid_tag(tag):
                writer.start(tag, {})

        for tag in reversed(tag_names):
            if is_valid_tag(tag):
                writer.end(tag)

    except Exception as e:
        raise AssertionError(f"start/end symmetry failed for tags {tag_names}: {e}") from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing inputs**:
- `tag_names=['0']` - tag starting with digit
- `tag_names=['\x1f']` - tag with control character
- Many others: `-tag`, `.tag`, `@attr`, `tag name`, `tag\ttab`, empty string

## Reproducing the Bug

```python
import sys
import tempfile
import shutil

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag

tmpdir = tempfile.mkdtemp()
try:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"

    assert is_valid_tag("0") == True
    writer.start("0", {})
except ValueError as e:
    print(f"Bug: is_valid_tag('0') returned True, but XML raised: {e}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

Output:
```
Bug: is_valid_tag('0') returned True, but XML raised: Invalid tag name '0'
```

## Why This Is A Bug

The function `is_valid_tag()` is supposed to determine if a name is valid for use as an XML tag in debug information. According to XML specifications, tag names must:
1. Start with a letter or underscore (not digit, hyphen, etc.)
2. Not contain control characters or NULL bytes
3. Not contain spaces or tabs
4. Not be empty

The current implementation only filters out the specific pattern ".DECIMAL" (for generator expression arguments), but allows through many other invalid XML tag names. This violates the function's contract and causes downstream crashes in `CythonDebugWriter.start()`, `end()`, and `add_entry()`.

## Fix

```diff
--- a/Cython/Debugger/DebugWriter.py
+++ b/Cython/Debugger/DebugWriter.py
@@ -14,6 +14,7 @@ from ..Compiler import Errors
 from ..Compiler.StringEncoding import EncodedString


+import re
 def is_valid_tag(name):
     """
     Names like '.0' are used internally for arguments
@@ -22,7 +23,16 @@ def is_valid_tag(name):

     See https://github.com/cython/cython/issues/5552
     """
+    # Convert to string if needed
+    name_str = str(name) if name else ""
+
+    # Check for empty string
+    if not name_str:
+        return False
+
     if isinstance(name, EncodedString):
         if name.startswith(".") and name[1:].isdecimal():
             return False
+
+    # Validate XML tag name rules: must start with letter/underscore, no control chars/spaces
+    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$', name_str)) and all(ord(c) >= 32 for c in name_str)
-    return True
```