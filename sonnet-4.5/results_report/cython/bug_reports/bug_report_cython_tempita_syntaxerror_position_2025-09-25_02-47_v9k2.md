# Bug Report: Cython.Tempita SyntaxError Missing Position Information

**Target**: `Cython.Tempita.Template._eval`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

SyntaxError exceptions in template expressions lack line/column position information, unlike other exceptions (NameError, etc.), making template debugging difficult.

## Property-Based Test

```python
@given(st.text(alphabet='0123456789+-*/', min_size=1, max_size=10))
def test_syntaxerror_includes_position(expr):
    assume('{{' not in expr and '}}' not in expr)

    content = f"Line 1\nLine 2\n{{{{{expr}}}}}"
    template = Template(content)

    try:
        template.substitute({})
    except SyntaxError as e:
        error_msg = str(e)
        assert 'line' in error_msg.lower() and 'column' in error_msg.lower()
```

**Failing input**: Any invalid Python expression like `/`, `**`, `+++`, etc.

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "Line 1\nLine 2\n{{/}}"
template = Template(content)

try:
    template.substitute({})
except SyntaxError as e:
    print(f"SyntaxError: {e}")

content2 = "Line 1\nLine 2\n{{undefined}}"
template2 = Template(content2)

try:
    template2.substitute({})
except NameError as e:
    print(f"NameError: {e}")
```

Output:
```
SyntaxError: invalid syntax in expression: /
NameError: name 'undefined' is not defined at line 3 column 3
```

## Why This Is A Bug

In the `_eval` method (lines 303-318):

```python
def _eval(self, code, ns, pos):
    __traceback_hide__ = True
    try:
        try:
            value = eval(code, self.default_namespace, ns)
        except SyntaxError as e:
            raise SyntaxError(
                'invalid syntax in expression: %s' % code)
        return value
    except Exception as e:
        if getattr(e, 'args', None):
            arg0 = e.args[0]
        else:
            arg0 = coerce_text(e)
        e.args = (self._add_line_info(arg0, pos),)
        raise
```

Lines 308-310 catch SyntaxError and raise a NEW SyntaxError with a custom message. This new exception bypasses the outer exception handler (lines 312-318) that adds position information via `_add_line_info`.

Other exceptions (NameError, ValueError, etc.) go through the outer handler and correctly get position info added.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -303,14 +303,11 @@ class Template:
 def _eval(self, code, ns, pos):
     __traceback_hide__ = True
     try:
-        try:
-            value = eval(code, self.default_namespace, ns)
-        except SyntaxError as e:
-            raise SyntaxError(
-                'invalid syntax in expression: %s' % code)
+        value = eval(code, self.default_namespace, ns)
         return value
+    except SyntaxError as e:
+        raise SyntaxError(
+            self._add_line_info('invalid syntax in expression: %s' % code, pos))
     except Exception as e:
         if getattr(e, 'args', None):
             arg0 = e.args[0]
```