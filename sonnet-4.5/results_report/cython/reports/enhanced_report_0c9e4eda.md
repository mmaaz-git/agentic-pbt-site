# Bug Report: Cython.Tempita TemplateDef Crashes When Using Keyword Arguments

**Target**: `Cython.Tempita._tempita.TemplateDef._parse_signature`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Cython Tempita template engine crashes with "TypeError: unhashable type: 'list'" when calling template-defined functions using keyword arguments due to a typo in the `_parse_signature` method.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Cython Tempita TemplateDef keyword arguments"""

from hypothesis import given, strategies as st
from Cython.Tempita import Template


@given(st.integers(), st.integers())
def test_templatedef_kwargs(x_val, y_val):
    template_str = """{{def myfunc(x, y)}}{{x}},{{y}}{{enddef}}{{myfunc(x=val_x, y=val_y)}}"""
    tmpl = Template(template_str)
    result = tmpl.substitute(val_x=x_val, val_y=y_val)
    expected = f"{x_val},{y_val}"
    assert result == expected


if __name__ == "__main__":
    test_templatedef_kwargs()
```

<details>

<summary>
**Failing input**: `x_val=0, y_val=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo_cython.py", line 18, in <module>
    test_templatedef_kwargs()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo_cython.py", line 9, in test_templatedef_kwargs
    def test_templatedef_kwargs(x_val, y_val):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo_cython.py", line 12, in test_templatedef_kwargs
    result = tmpl.substitute(val_x=x_val, val_y=y_val)
  File "Cython/Tempita/_tempita.py", line 186, in Cython.Tempita._tempita.Template.substitute
  File "Cython/Tempita/_tempita.py", line 197, in Cython.Tempita._tempita.Template._interpret
  File "Cython/Tempita/_tempita.py", line 225, in Cython.Tempita._tempita.Template._interpret_codes
  File "Cython/Tempita/_tempita.py", line 245, in Cython.Tempita._tempita.Template._interpret_code
  File "Cython/Tempita/_tempita.py", line 318, in Cython.Tempita._tempita.Template._eval
  File "Cython/Tempita/_tempita.py", line 307, in Cython.Tempita._tempita.Template._eval
  File "<string>", line 1, in <module>
  File "Cython/Tempita/_tempita.py", line 443, in Cython.Tempita._tempita.TemplateDef.__call__
  File "Cython/Tempita/_tempita.py", line 469, in Cython.Tempita._tempita.TemplateDef._parse_signature
TypeError: unhashable type: 'list' at line 1 column 44
Falsifying example: test_templatedef_kwargs(
    x_val=0,
    y_val=0,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython Tempita TemplateDef keyword argument crash"""

from Cython.Tempita import Template

# Create a template with a function definition that accepts keyword arguments
template_str = """{{def myfunc(x, y)}}{{x}},{{y}}{{enddef}}{{myfunc(x=1, y=2)}}"""

try:
    tmpl = Template(template_str)
    result = tmpl.substitute()
    print(f"Success: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError: unhashable type: 'list' at line 1 column 44
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/repo_cython.py", line 11, in <module>
    result = tmpl.substitute()
  File "Cython/Tempita/_tempita.py", line 186, in Cython.Tempita._tempita.Template.substitute
  File "Cython/Tempita/_tempita.py", line 197, in Cython.Tempita._tempita.Template._interpret
  File "Cython/Tempita/_tempita.py", line 225, in Cython.Tempita._tempita.Template._interpret_codes
  File "Cython/Tempita/_tempita.py", line 245, in Cython.Tempita._tempita.Template._interpret_code
  File "Cython/Tempita/_tempita.py", line 318, in Cython.Tempita._tempita.Template._eval
  File "Cython/Tempita/_tempita.py", line 307, in Cython.Tempita._tempita.Template._eval
  File "<string>", line 1, in <module>
  File "Cython/Tempita/_tempita.py", line 443, in Cython.Tempita._tempita.TemplateDef.__call__
  File "Cython/Tempita/_tempita.py", line 469, in Cython.Tempita._tempita.TemplateDef._parse_signature
TypeError: unhashable type: 'list' at line 1 column 44
Error type: TypeError
Error message: unhashable type: 'list' at line 1 column 44
```
</details>

## Why This Is A Bug

The Cython Tempita template engine supports the `{{def}}` directive for defining reusable template functions. The implementation in `_tempita.py` clearly intends to support keyword arguments:

1. The `parse_signature` function at line 938 parses Python-style function signatures including default arguments, *args, and **kwargs
2. The `_parse_signature` method at line 460 has explicit code branches to handle keyword arguments (lines 464-471)
3. The error occurs on line 469 where `values[sig_args] = value` attempts to use `sig_args` (a list like `['x', 'y']`) as a dictionary key instead of the variable `name` (a string like `'x'`)

This is a clear typo that makes keyword arguments completely unusable in template-defined functions, forcing users to only use positional arguments. The bug violates the expected behavior that template functions should work similarly to Python functions.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py` at line 469.

The problematic code section:
```python
def _parse_signature(self, args, kw):
    values = {}
    sig_args, var_args, var_kw, defaults = self._func_signature
    extra_kw = {}
    for name, value in kw.items():
        if not var_kw and name not in sig_args:
            raise TypeError(
                'Unexpected argument %s' % name)
        if name in sig_args:
            values[sig_args] = value  # BUG: Should be values[name] = value
        else:
            extra_kw[name] = value
```

The `sig_args` variable contains a list of all signature arguments (e.g., `['x', 'y']`), while `name` contains the current argument being processed (e.g., `'x'`). Lists are unhashable and cannot be used as dictionary keys in Python, causing the TypeError.

Workaround: Users can work around this bug by using only positional arguments when calling template-defined functions: `{{myfunc(1, 2)}}` instead of `{{myfunc(x=1, y=2)}}`.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -466,7 +466,7 @@ class TemplateDef:
                 raise TypeError(
                     'Unexpected argument %s' % name)
             if name in sig_args:
-                values[sig_args] = value
+                values[name] = value
             else:
                 extra_kw[name] = value
         args = list(args)
```