# Bug Report: Cython.Tempita fill_command Incorrect py: Prefix Removal

**Target**: `Cython.Tempita._tempita.fill_command`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fill_command` function incorrectly extracts variable names from `py:` prefixed arguments, keeping only the prefix "py:" instead of removing it, causing template variables to be set incorrectly.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import string
from Cython.Tempita._tempita import fill_command

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier))
def test_fill_command_py_prefix_removal(var_name):
    args = ['-', f'py:{var_name}=42']

    # Mock stdout to capture output
    import io
    import sys
    old_stdin = sys.stdin
    old_stdout = sys.stdout

    try:
        sys.stdin = io.StringIO(f"{{{{{var_name}}}}}")
        sys.stdout = io.StringIO()

        fill_command(args)
        result = sys.stdout.getvalue()

        assert '42' in result, f"Variable {var_name} should be set to 42"
        assert 'py:' not in result, "Variable name should not include 'py:' prefix"
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
```

**Failing input**: Any command-line argument like `py:x=42`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import io
from Cython.Tempita._tempita import fill_command

old_stdin = sys.stdin
old_stdout = sys.stdout

try:
    sys.stdin = io.StringIO("{{x}}")
    sys.stdout = io.StringIO()

    args = ['-', 'py:x=42']
    fill_command(args)

    result = sys.stdout.getvalue()
    print(f"Result: {result!r}")
finally:
    sys.stdin = old_stdin
    sys.stdout = old_stdout
```

**Expected**: Output should be `"42"` (variable `x` set to integer 42)

**Actual**: Error or incorrect variable name (variable `py:` set instead of `x`)

## Why This Is A Bug

In `Cython/Tempita/_tempita.py` line 1073:

```python
for value in args:
    if '=' not in value:
        print('Bad argument: %r' % value)
        sys.exit(2)
    name, value = value.split('=', 1)
    if name.startswith('py:'):
        name = name[:3]  # Line 1073: BUG! Should be name[3:] to REMOVE prefix
        value = eval(value)
    vars[name] = value
```

Line 1073 uses `name[:3]` which keeps the FIRST 3 characters ("py:"), when it should use `name[3:]` to REMOVE the first 3 characters. This means:
- Input: `py:x=42`
- After split: `name='py:x'`, `value='42'`
- After `name[:3]`: `name='py:'` ❌ (should be `'x'`)
- Result: Variable `'py:'` is set to 42 instead of variable `'x'`

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -1070,7 +1070,7 @@ def fill_command(args=None):
         name, value = value.split('=', 1)
         if name.startswith('py:'):
-            name = name[:3]
+            name = name[3:]
             value = eval(value)
         vars[name] = value
```