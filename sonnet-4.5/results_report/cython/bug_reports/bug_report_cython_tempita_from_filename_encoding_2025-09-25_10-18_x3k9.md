# Bug Report: Cython.Tempita Template.from_filename Encoding Missing

**Target**: `Cython.Tempita._tempita.Template.from_filename`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Template.from_filename() crashes with TypeError when the encoding parameter is not specified, despite having a default_encoding class attribute that should be used automatically.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os
from Cython.Tempita import Template

@given(st.text(alphabet=string.printable, max_size=200))
@settings(max_examples=100)
def test_from_filename_without_encoding(content):
    if '{{' in content or '}}' in content:
        return

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
        f.write(content)
        filename = f.name

    try:
        template = Template.from_filename(filename)
        result = template.substitute({})
        assert result == content
    finally:
        os.unlink(filename)
```

**Failing input**: Any non-empty file content

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import tempfile
import os
from Cython.Tempita import Template

with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
    f.write("Hello World")
    filename = f.name

try:
    template = Template.from_filename(filename)
    print("Template created successfully")
except TypeError as e:
    print(f"TypeError: {e}")
finally:
    os.unlink(filename)
```

Output:
```
TypeError: cannot use a string pattern on a bytes-like object
```

## Why This Is A Bug

In `Template.from_filename` at lines 153-160:

```python
def from_filename(cls, filename, namespace=None, encoding=None,
                  default_inherit=None, get_template=get_file_template):
    with open(filename, 'rb') as f:
        c = f.read()
    if encoding:
        c = c.decode(encoding)
    return cls(content=c, name=filename, namespace=namespace,
               default_inherit=default_inherit, get_template=get_template)
```

The function:
1. Opens the file in binary mode ('rb')
2. Reads content as bytes
3. Only decodes if encoding is explicitly provided
4. Passes bytes to Template.__init__ when encoding=None

Then Template.__init__ calls `parse(content, ...)` at line 145, which calls `lex(s, ...)` at line 739. The lex() function uses `re.compile(...).finditer(s)` at line 581, which fails on bytes objects in Python 3.

The Template class has `default_encoding = 'utf8'` at line 96, which is used for encoding/decoding during template execution (lines 345-370), but is NOT used in from_filename() for the initial file reading.

The encoding parameter has a default value of None, which strongly suggests it should be optional. The existence of default_encoding also implies automatic decoding should work.

This breaks the documented API - users expect from_filename() to "just work" without specifying encoding for UTF-8 files.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -154,8 +154,10 @@ class Template:
                       default_inherit=None, get_template=get_file_template):
         with open(filename, 'rb') as f:
             c = f.read()
-        if encoding:
-            c = c.decode(encoding)
+        if encoding is None:
+            encoding = cls.default_encoding
+        if encoding:
+            c = c.decode(encoding)
         return cls(content=c, name=filename, namespace=namespace,
                    default_inherit=default_inherit, get_template=get_template)
```