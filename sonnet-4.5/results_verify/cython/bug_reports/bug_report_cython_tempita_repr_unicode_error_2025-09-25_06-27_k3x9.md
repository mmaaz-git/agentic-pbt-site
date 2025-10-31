# Bug Report: Cython.Tempita Template._repr UnicodeDecodeError Construction

**Target**: `Cython.Tempita._tempita.Template._repr`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Template._repr` method incorrectly constructs `UnicodeDecodeError` and `UnicodeEncodeError` exceptions with a single string argument instead of the required 5 arguments, causing a TypeError when these error paths are triggered.

## Property-Based Test

```python
@given(st.binary(min_size=1, max_size=10))
@settings(max_examples=100)
def test_template_bytes_value_without_encoding(byte_value):
    content = "{{x}}"
    template = Template(content)

    try:
        result = template.substitute({'x': byte_value})
        assert False, "Should raise an encoding error"
    except (UnicodeDecodeError, TypeError):
        pass
```

**Failing input**: Any bytes value when template has no default_encoding, e.g., `b'\xff'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{x}}"
template = Template(content)

try:
    result = template.substitute({'x': b'\xff'})
except TypeError as e:
    print(f"TypeError raised: {e}")
    print(f"Expected: UnicodeDecodeError with proper message")
    print(f"Actual: TypeError about UnicodeDecodeError constructor")
```

## Why This Is A Bug

Lines 353-355 construct a UnicodeDecodeError with a single formatted string argument:
```python
raise UnicodeDecodeError(
    'Cannot decode bytes value %r into unicode '
    '(no default_encoding provided)' % value)
```

However, the UnicodeDecodeError constructor requires exactly 5 arguments:
```python
UnicodeDecodeError(encoding, object, start, end, reason)
```

Similarly, lines 367-369 have the same issue with UnicodeEncodeError:
```python
raise UnicodeEncodeError(
    'Cannot encode unicode value %r into bytes '
    '(no default_encoding provided)' % value)
```

This causes a TypeError like: "UnicodeDecodeError() takes exactly 5 arguments (1 given)" instead of raising the intended error message.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -350,8 +350,8 @@ class Template:
         else:
             if self._unicode and isinstance(value, bytes):
                 if not self.default_encoding:
-                    raise UnicodeDecodeError(
-                        'Cannot decode bytes value %r into unicode '
-                        '(no default_encoding provided)' % value)
+                    raise ValueError(
+                        'Cannot decode bytes value %r into unicode '
+                        '(no default_encoding provided)' % value)
                 try:
                     value = value.decode(self.default_encoding)
                 except UnicodeDecodeError as e:
@@ -364,8 +364,8 @@ class Template:
                         e.reason + ' in string %r' % value)
             elif not self._unicode and isinstance(value, str):
                 if not self.default_encoding:
-                    raise UnicodeEncodeError(
-                        'Cannot encode unicode value %r into bytes '
-                        '(no default_encoding provided)' % value)
+                    raise ValueError(
+                        'Cannot encode unicode value %r into bytes '
+                        '(no default_encoding provided)' % value)
                 value = value.encode(self.default_encoding)
             return value
```