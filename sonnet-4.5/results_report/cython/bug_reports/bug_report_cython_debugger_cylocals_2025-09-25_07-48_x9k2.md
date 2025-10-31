# Bug Report: Cython.Debugger CyLocals Empty Dict Crash

**Target**: `Cython.Debugger.libcython.CyLocals.invoke`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `CyLocals.invoke` method crashes with a `ValueError` when a Cython function has no local variables, due to calling `max()` on an empty dictionary without a guard.

## Property-Based Test

```python
from hypothesis import given, strategies as st


sortkey = lambda item: item[0].lower()


def cy_locals_invoke_logic(local_cython_vars):
    max_name_length = len(max(local_cython_vars, key=len))
    for name, cyvar in sorted(local_cython_vars.items(), key=sortkey):
        pass
    return max_name_length


@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_cy_locals_with_various_dicts(local_vars):
    if len(local_vars) == 0:
        try:
            cy_locals_invoke_logic(local_vars)
            assert False, "Should have raised ValueError for empty dict"
        except ValueError:
            pass
    else:
        result = cy_locals_invoke_logic(local_vars)
        assert result >= 0
```

**Failing input**: `local_vars = {}`

## Reproducing the Bug

```python
local_cython_vars = {}

max_name_length = len(max(local_cython_vars, key=len))
```

Running this code produces:
```
ValueError: max() arg is an empty sequence
```

## Why This Is A Bug

1. **Inconsistency**: The sibling method `CyGlobals.invoke` (libcython.py:1288-1291) properly guards against empty dictionaries, but `CyLocals.invoke` does not.

2. **Real scenario**: A Cython function with no local variables is valid (e.g., a simple `pass` function or one that only uses globals).

3. **Expected behavior**: The `cy locals` command should handle functions with no locals gracefully, printing nothing or an empty result, not crashing.

## Fix

```diff
--- a/Cython/Debugger/libcython.py
+++ b/Cython/Debugger/libcython.py
@@ -1259,7 +1259,9 @@ class CyLocals(CythonCommand):
             return

         local_cython_vars = cython_function.locals
-        max_name_length = len(max(local_cython_vars, key=len))
+        max_name_length = 0
+        if local_cython_vars:
+            max_name_length = len(max(local_cython_vars, key=len))
         for name, cyvar in sorted(local_cython_vars.items(), key=sortkey):
             if self.is_initialized(self.get_cython_function(), cyvar.name):
                 value = gdb.parse_and_eval(cyvar.cname)
```