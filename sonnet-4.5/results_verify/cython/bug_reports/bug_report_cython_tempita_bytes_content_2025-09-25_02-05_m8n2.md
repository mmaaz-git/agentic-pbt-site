# Bug Report: Cython.Tempita Bytes Content Type Error

**Target**: `Cython.Tempita._tempita.lex`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Template constructor accepts bytes content and sets the `_unicode` flag accordingly, but the `lex()` function crashes when trying to parse bytes content because it uses a string regex pattern.

## Property-Based Test

```python
@given(st.text(min_size=1, max_size=100))
def test_template_bytes_content_handling(value):
    assume('\x00' not in value)

    content_bytes = value.encode('utf-8')
    template = Template(content_bytes)

    result = template.substitute({})
    assert isinstance(result, bytes) or isinstance(result, str)
```

**Failing input**: Any bytes content, e.g., `b'Hello {{name}}'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

bytes_content = b"Hello {{name}}"
template = Template(bytes_content)
result = template.substitute({'name': 'World'})
print(result)
```

**Output**: `TypeError: cannot use a string pattern on a bytes-like object` at `_tempita.py:581`

## Why This Is A Bug

The Template class explicitly supports bytes content via the `_unicode` flag (line 125):
```python
self._unicode = isinstance(content, str)
```

This suggests bytes content should be supported. However, the `lex()` function (called from `parse()` at line 739) uses a string regex pattern (lines 579-580) and applies it to the bytes content at line 581:

```python
token_re = re.compile(r'%s|%s' % (re.escape(delimiters[0]),
                                  re.escape(delimiters[1])))
for match in token_re.finditer(s):  # Crashes if s is bytes
```

Python's `re.finditer()` requires the pattern and target to be the same type (both str or both bytes). Since the pattern is always a string, it fails when `s` is bytes.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -576,8 +576,14 @@ def lex(s, name=None, trim_whitespace=True, line_offset=0, delimiters=None):
     last = 0
     last_pos = (line_offset + 1, 1)

-    token_re = re.compile(r'%s|%s' % (re.escape(delimiters[0]),
-                                      re.escape(delimiters[1])))
+    # Handle both str and bytes content
+    if isinstance(s, bytes):
+        pattern = b'%s|%s' % (re.escape(delimiters[0].encode('utf-8')),
+                              re.escape(delimiters[1].encode('utf-8')))
+        token_re = re.compile(pattern)
+    else:
+        token_re = re.compile(r'%s|%s' % (re.escape(delimiters[0]),
+                                          re.escape(delimiters[1])))
     for match in token_re.finditer(s):
         expr = match.group(0)
         pos = find_position(s, match.end(), last, last_pos)
```