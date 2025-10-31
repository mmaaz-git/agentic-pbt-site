# Bug Report: Cython.Tempita TemplateDef Keyword Arguments Crash

**Target**: `Cython.Tempita._tempita.TemplateDef._parse_signature`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling a template-defined function with keyword arguments, the code crashes with "TypeError: unhashable type: 'list'" due to using a list as a dictionary key.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita import Template


@given(st.integers(), st.integers())
def test_templatedef_kwargs(x_val, y_val):
    template_str = """{{def myfunc(x, y)}}{{x}},{{y}}{{enddef}}{{myfunc(x=val_x, y=val_y)}}"""
    tmpl = Template(template_str)
    result = tmpl.substitute(val_x=x_val, val_y=y_val)
    expected = f"{x_val},{y_val}"
    assert result == expected
```

**Failing input**: Any values for `x_val` and `y_val`

## Reproducing the Bug

```python
from Cython.Tempita import Template

template_str = """{{def myfunc(x, y)}}{{x}},{{y}}{{enddef}}{{myfunc(x=1, y=2)}}"""
tmpl = Template(template_str)
result = tmpl.substitute()
```

Output:
```
TypeError: unhashable type: 'list'
```

## Why This Is A Bug

The bug is on line 469 of `_tempita.py` in the `_parse_signature` method. The code attempts to use `sig_args` (a list) as a dictionary key instead of the intended `name` variable:

```python
values[sig_args] = value  # BUG: sig_args is a list!
```

This makes it impossible to call template-defined functions with keyword arguments, which is a core feature of the templating system.

## Fix

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