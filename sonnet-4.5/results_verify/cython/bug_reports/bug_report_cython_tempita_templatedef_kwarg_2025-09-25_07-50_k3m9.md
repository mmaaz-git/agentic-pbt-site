# Bug Report: Cython.Tempita TemplateDef Keyword Argument Assignment

**Target**: `Cython.Tempita._tempita.TemplateDef._parse_signature`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling a template function with keyword arguments, the code attempts to use a list as a dictionary key, causing a `TypeError: unhashable type: 'list'`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from Cython.Tempita import Template

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
       st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
       st.text(max_size=20),
       st.text(max_size=20))
@settings(max_examples=500)
def test_templatedef_with_keyword_args(arg_name, kwarg_name, pos_val, kw_val):
    if not arg_name.isidentifier() or not kwarg_name.isidentifier():
        return
    if arg_name == kwarg_name:
        return
    assume('{{' not in pos_val and '}}' not in pos_val)
    assume('{{' not in kw_val and '}}' not in kw_val)

    template_content = f"""
{{{{def myfunc({arg_name}, {kwarg_name}='default')}}}}
{arg_name}={{{{{arg_name}}}}},{kwarg_name}={{{{{kwarg_name}}}}}
{{{{enddef}}}}
{{{{myfunc('{pos_val}', {kwarg_name}='{kw_val}')}}}}
"""

    t = Template(template_content)
    result = t.substitute({})

    assert f'{arg_name}={pos_val}' in result
    assert f'{kwarg_name}={kw_val}' in result
```

**Failing input**: Any template function called with keyword arguments, e.g., `arg_name='_'`, `kwarg_name='__'`, `pos_val=''`, `kw_val=''`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

template_content = """
{{def myfunc(arg1, kwarg1='default')}}
result: {{arg1}}, {{kwarg1}}
{{enddef}}

{{myfunc('value1', kwarg1='value2')}}
"""

t = Template(template_content)
result = t.substitute({})
print(result)
```

This raises: `TypeError: unhashable type: 'list' at line 6 column 3`

## Why This Is A Bug

The `_parse_signature` method at line 469 attempts to assign a value using `sig_args` (a list) as a dictionary key:

```python
values[sig_args] = value  # Line 469 - WRONG
```

Lists are unhashable and cannot be used as dictionary keys in Python. The variable `name` (from line 464) should be used instead, matching the pattern at line 479 for positional arguments.

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