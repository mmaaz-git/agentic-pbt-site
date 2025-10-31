# Bug Report: Cython.Tempita Template.from_filename Crashes Without Encoding Parameter

**Target**: `Cython.Tempita._tempita.Template.from_filename`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Template.from_filename()` method crashes with a TypeError when called without the optional `encoding` parameter, despite having a default value of `None` in its signature and a `default_encoding` class attribute that should provide a fallback.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import tempfile
import os
import string
from Cython.Tempita import Template

@given(st.text(alphabet=string.printable, max_size=200))
@settings(max_examples=100)
def test_from_filename_without_encoding(content):
    # Skip content with template delimiters to avoid template parsing issues
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

if __name__ == "__main__":
    # Run the test
    test_from_filename_without_encoding()
```

<details>

<summary>
**Failing input**: `''` (empty string)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 30, in <module>
    test_from_filename_without_encoding()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 11, in test_from_filename_without_encoding
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 22, in test_from_filename_without_encoding
    template = Template.from_filename(filename)
  File "Cython/Tempita/_tempita.py", line 159, in Cython.Tempita._tempita.Template.from_filename
  File "Cython/Tempita/_tempita.py", line 145, in Cython.Tempita._tempita.Template.__init__
  File "Cython/Tempita/_tempita.py", line 739, in Cython.Tempita._tempita.parse
  File "Cython/Tempita/_tempita.py", line 581, in Cython.Tempita._tempita.lex
TypeError: cannot use a string pattern on a bytes-like object
Falsifying example: test_from_filename_without_encoding(
    content='',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import tempfile
import os
from Cython.Tempita import Template

# Create a temporary file with simple content
with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
    f.write("Hello World")
    filename = f.name

try:
    # Try to create a template without specifying encoding
    template = Template.from_filename(filename)
    result = template.substitute({})
    print(f"Success: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(filename)
```

<details>

<summary>
TypeError: cannot use a string pattern on a bytes-like object
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 15, in <module>
    template = Template.from_filename(filename)
  File "Cython/Tempita/_tempita.py", line 159, in Cython.Tempita._tempita.Template.from_filename
  File "Cython/Tempita/_tempita.py", line 145, in Cython.Tempita._tempita.Template.__init__
  File "Cython/Tempita/_tempita.py", line 739, in Cython.Tempita._tempita.parse
  File "Cython/Tempita/_tempita.py", line 581, in Cython.Tempita._tempita.lex
TypeError: cannot use a string pattern on a bytes-like object
TypeError: cannot use a string pattern on a bytes-like object
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Method signature implies optional parameter**: The `from_filename` method at line 153 defines `encoding=None` as a default parameter, strongly suggesting it should be optional and work without being explicitly specified.

2. **Class has a default encoding**: The Template class defines `default_encoding = 'utf8'` at line 96, which appears designed to provide a fallback when no encoding is specified, but this is never used in `from_filename()`.

3. **Python 3 bytes vs strings incompatibility**: The method reads files in binary mode (`'rb'` at line 155) but only decodes to a string if encoding is explicitly provided (lines 157-158). When encoding is None, raw bytes are passed to `Template.__init__`, which eventually calls `lex()` at line 581. The `lex()` function uses `re.compile().finditer()` on the content, which fails on bytes objects in Python 3.

4. **Complete failure for basic usage**: The method is completely unusable without specifying encoding - even for empty files or simple ASCII content. This breaks the most basic use case of `Template.from_filename(filepath)`.

5. **No documentation warning**: There is no docstring or documentation indicating that the encoding parameter is required, despite the method signature suggesting otherwise.

## Relevant Context

The bug appears to be a Python 2 to Python 3 migration issue where bytes/string handling wasn't properly updated. The pattern suggests this code may have worked in Python 2 where strings and bytes were more interchangeable.

Code locations:
- Method definition: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py:153-160`
- Class default_encoding: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py:96`
- Failure point (lex function): `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py:581`

Workaround: Users can work around this by explicitly specifying `encoding='utf-8'` when calling `Template.from_filename()`.

## Proposed Fix

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