# Bug Report: Cython.Tempita fill_command py: Prefix Parsing

**Target**: `Cython.Tempita._tempita.fill_command`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fill_command` function incorrectly parses `py:` prefixed arguments, keeping only the prefix "py:" instead of the actual variable name, causing all Python-evaluated arguments to be stored under the same key "py:".

## Property-Based Test

```python
@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=10).filter(str.isidentifier))
@settings(max_examples=100)
def test_fill_command_py_prefix_strips_prefix(var_name):
    arg_string = f"py:{var_name}"

    name = arg_string
    if name.startswith('py:'):
        parsed_name = name[:3]

    expected_name = var_name
    actual_name = parsed_name

    assert actual_name == expected_name, f"Variable name should be {expected_name!r}, got {actual_name!r}"
```

**Failing input**: Any argument like `py:x=42` or `py:my_var=123`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

arg_string = "py:my_var"

name = arg_string
if name.startswith('py:'):
    name = name[:3]

print(f"Input argument: py:my_var=42")
print(f"Expected variable name: 'my_var'")
print(f"Actual variable name: {name!r}")
print(f"Bug: All py: arguments are stored as vars['py:'] instead of vars['my_var']")
```

## Why This Is A Bug

Line 1073 in `Cython/Tempita/_tempita.py` contains `name = name[:3]` which keeps the first 3 characters ("py:") instead of removing them. The intent is clearly to strip the "py:" prefix to get the actual variable name.

The docstring at line 1032 states: "Use py:arg=value to set a Python value", implying that `py:x=42` should set variable `x` (not `py:`) to the evaluated value 42.

This bug causes:
1. All Python-evaluated arguments to overwrite each other (all stored as `vars['py:']`)
2. The actual variable name to be lost
3. Templates expecting variables like `{{my_var}}` to fail even when `py:my_var=value` was passed

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -1070,7 +1070,7 @@ def fill_command(args=None):
             sys.exit(2)
         name, value = value.split('=', 1)
         if name.startswith('py:'):
-            name = name[:3]
+            name = name[3:]
             value = eval(value)
         vars[name] = value
     if template_name == '-':
```