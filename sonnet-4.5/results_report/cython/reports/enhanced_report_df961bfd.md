# Bug Report: Cython.Tempita._tempita.lex TypeError with Bytes Content

**Target**: `Cython.Tempita._tempita.lex`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Template class accepts bytes content and tracks it via the `_unicode` flag, but crashes in the `lex()` function when parsing bytes because it uses a string regex pattern on bytes content.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from Cython.Tempita import Template

@given(st.text(min_size=1, max_size=100))
def test_template_bytes_content_handling(value):
    assume('\x00' not in value)

    content_bytes = value.encode('utf-8')
    template = Template(content_bytes)

    result = template.substitute({})
    assert isinstance(result, bytes) or isinstance(result, str)

# Run the test
test_template_bytes_content_handling()
```

<details>

<summary>
**Failing input**: `value='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 18, in <module>
    test_template_bytes_content_handling()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 8, in test_template_bytes_content_handling
    def test_template_bytes_content_handling(value):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 12, in test_template_bytes_content_handling
    template = Template(content_bytes)
  File "Cython/Tempita/_tempita.py", line 145, in Cython.Tempita._tempita.Template.__init__
  File "Cython/Tempita/_tempita.py", line 739, in Cython.Tempita._tempita.parse
  File "Cython/Tempita/_tempita.py", line 581, in Cython.Tempita._tempita.lex
TypeError: cannot use a string pattern on a bytes-like object
Falsifying example: test_template_bytes_content_handling(
    value='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test with bytes content
bytes_content = b"Hello {{name}}"
template = Template(bytes_content)
result = template.substitute({'name': 'World'})
print(result)
```

<details>

<summary>
TypeError: cannot use a string pattern on a bytes-like object
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 8, in <module>
    template = Template(bytes_content)
  File "Cython/Tempita/_tempita.py", line 145, in Cython.Tempita._tempita.Template.__init__
  File "Cython/Tempita/_tempita.py", line 739, in Cython.Tempita._tempita.parse
  File "Cython/Tempita/_tempita.py", line 581, in Cython.Tempita._tempita.lex
TypeError: cannot use a string pattern on a bytes-like object
```
</details>

## Why This Is A Bug

The Template class is explicitly designed to support both bytes and string content, as evidenced by multiple implementation details:

1. **Line 125**: The constructor sets `self._unicode = isinstance(content, str)` to track whether the content is unicode or bytes. This flag would be unnecessary if bytes weren't intended to be supported.

2. **Lines 336-346**: The `_repr()` method contains explicit branching logic based on the `_unicode` flag, with separate handling for bytes vs string content.

3. **Lines 351-371**: Extensive code for converting between bytes and unicode, including encoding/decoding with `self.default_encoding`.

4. **Line 155**: The `from_filename()` method reads files in binary mode (`'rb'`), indicating that bytes content is expected to be processable.

5. **Line 50**: The `coerce_text()` helper explicitly returns `bytes(v)` for non-string values.

Despite this infrastructure, the `lex()` function (lines 579-581) always creates a string regex pattern using `r'%s|%s'`, then applies it to the content which may be bytes. Python's regex engine requires the pattern and target string to be the same type - both str or both bytes. This type mismatch causes the crash.

The bug represents an incomplete implementation where bytes support was added throughout the codebase except in the critical lexer function.

## Relevant Context

The Cython.Tempita module is a templating engine included with Cython. While the module docstring describes it as "a small templating language for text substitution," the implementation clearly attempts to support both text (str) and binary (bytes) content.

This bug prevents legitimate use cases such as:
- Processing templates from binary files with unknown or mixed encodings
- Working with templates containing binary data
- Processing non-UTF8 encoded template files

A workaround exists: users can decode bytes to strings before passing to Template, but this requires knowing the encoding and may fail for binary content.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -576,8 +576,17 @@ def lex(s, name=None, trim_whitespace=True, line_offset=0, delimiters=None):
     last = 0
     last_pos = (line_offset + 1, 1)

-    token_re = re.compile(r'%s|%s' % (re.escape(delimiters[0]),
-                                      re.escape(delimiters[1])))
+    # Handle both str and bytes content
+    if isinstance(s, bytes):
+        # Create bytes pattern for bytes content
+        delim_start = delimiters[0].encode('utf-8') if isinstance(delimiters[0], str) else delimiters[0]
+        delim_end = delimiters[1].encode('utf-8') if isinstance(delimiters[1], str) else delimiters[1]
+        pattern = b'%s|%s' % (re.escape(delim_start),
+                              re.escape(delim_end))
+        token_re = re.compile(pattern)
+    else:
+        token_re = re.compile(r'%s|%s' % (re.escape(delimiters[0]),
+                                          re.escape(delimiters[1])))
     for match in token_re.finditer(s):
         expr = match.group(0)
         pos = find_position(s, match.end(), last, last_pos)
```