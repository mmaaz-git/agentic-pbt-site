# Bug Report: Flask get_root_path Documentation Violation

**Target**: `flask.helpers.get_root_path`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_root_path` function violates its documented contract by raising `RuntimeError` for built-in modules without a `__file__` attribute, instead of returning the current working directory as the docstring claims.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from flask.helpers import get_root_path

@given(st.sampled_from(['sys', 'builtins', 'marshal']))
def test_get_root_path_should_not_raise_for_builtin_modules(module_name):
    result = get_root_path(module_name)
    assert isinstance(result, str)
```

**Failing input**: `'sys'`

## Reproducing the Bug

```python
from flask.helpers import get_root_path

result = get_root_path('sys')
```

**Expected behavior** (according to docstring): Returns the current working directory

**Actual behavior**: Raises `RuntimeError: No root path can be found for the provided module 'sys'...`

## Why This Is A Bug

The function's docstring explicitly states:

> "Find the root path of a package, or the path that contains a module. **If it cannot be found, returns the current working directory.**"

However, when called with a built-in module like `sys` that has no `__file__` attribute, the function raises a `RuntimeError` instead of returning `os.getcwd()` as documented.

The docstring establishes a clear contract that the function should never raise an exception for modules where the root path cannot be determined - it should fall back to returning the current working directory. This contract violation could cause unexpected crashes in user code.

Interestingly, the function does correctly return `os.getcwd()` for truly nonexistent modules, but fails for existing built-in modules.

## Fix

```diff
--- a/flask/helpers.py
+++ b/flask/helpers.py
@@ -617,14 +617,8 @@ def get_root_path(import_name: str) -> str:
         filepath = getattr(mod, "__file__", None)

         # If we don't have a file path it might be because it is a
-        # namespace package. In this case pick the root path from the
-        # first module that is contained in the package.
+        # namespace package or built-in module. Fall back to cwd.
         if filepath is None:
-            raise RuntimeError(
-                "No root path can be found for the provided module"
-                f" {import_name!r}. This can happen because the module"
-                " came from an import hook that does not provide file"
-                " name information or because it's a namespace package."
-                " In this case the root path needs to be explicitly"
-                " provided."
-            )
+            return os.getcwd()

     # filepath is import_name.py for a module, or __init__.py for a package.