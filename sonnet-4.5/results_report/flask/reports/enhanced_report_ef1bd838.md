# Bug Report: Flask get_root_path Raises Exception for Built-in Modules Despite Documentation

**Target**: `flask.helpers.get_root_path`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_root_path` function violates its documented contract by raising `RuntimeError` for built-in modules instead of returning the current working directory as promised in the docstring.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from flask.helpers import get_root_path

@given(st.sampled_from(['sys', 'builtins', 'marshal']))
def test_get_root_path_should_not_raise_for_builtin_modules(module_name):
    result = get_root_path(module_name)
    assert isinstance(result, str)
```

<details>

<summary>
**Failing input**: `'sys'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/12
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_get_root_path_should_not_raise_for_builtin_modules FAILED

=================================== FAILURES ===================================
___________ test_get_root_path_should_not_raise_for_builtin_modules ____________

    @given(st.sampled_from(['sys', 'builtins', 'marshal']))
>   def test_get_root_path_should_not_raise_for_builtin_modules(module_name):
                   ^^^

hypo.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
hypo.py:6: in test_get_root_path_should_not_raise_for_builtin_modules
    result = get_root_path(module_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

import_name = 'sys'

    def get_root_path(import_name: str) -> str:
        """Find the root path of a package, or the path that contains a
        module. If it cannot be found, returns the current working
        directory.

        Not to be confused with the value returned by :func:`find_package`.

        :meta private:
        """
        # Module already imported and has a file attribute. Use that first.
        mod = sys.modules.get(import_name)

        if mod is not None and hasattr(mod, "__file__") and mod.__file__ is not None:
            return os.path.dirname(os.path.abspath(mod.__file__))

        # Next attempt: check the loader.
        try:
            spec = importlib.util.find_spec(import_name)

            if spec is None:
                raise ValueError
        except (ImportError, ValueError):
            loader = None
        else:
            loader = spec.loader

        # Loader does not exist or we're referring to an unloaded main
        # module or a main module without path (interactive sessions), go
        # with the current working directory.
        if loader is None:
            return os.getcwd()

        if hasattr(loader, "get_filename"):
            filepath = loader.get_filename(import_name)  # pyright: ignore
        else:
            # Fall back to imports.
            __import__(import_name)
            mod = sys.modules[import_name]
            filepath = getattr(mod, "__file__", None)

            # If we don't have a file path it might be because it is a
            # namespace package. In this case pick the root path from the
            # first module that is contained in the package.
            if filepath is None:
>               raise RuntimeError(
                    "No root path can be found for the provided module"
                    f" {import_name!r}. This can happen because the module"
                    " came from an import hook that does not provide file"
                    " name information or because it's a namespace package."
                    " In this case the root path needs to be explicitly"
                    " provided."
                )
E               RuntimeError: No root path can be found for the provided module 'sys'. This can happen because the module came from an import hook that does not provide file name information or because it's a namespace package. In this case the root path needs to be explicitly provided.
E               Falsifying example: test_get_root_path_should_not_raise_for_builtin_modules(
E                   module_name='sys',
E               )

/home/npc/miniconda/lib/python3.13/site-packages/flask/helpers.py:614: RuntimeError
=========================== short test summary info ============================
FAILED hypo.py::test_get_root_path_should_not_raise_for_builtin_modules - Run...
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
============================== 1 failed in 0.19s ===============================
```
</details>

## Reproducing the Bug

```python
from flask.helpers import get_root_path

# Test with built-in module 'sys'
try:
    result = get_root_path('sys')
    print(f"Result for 'sys': {result}")
except Exception as e:
    print(f"Error for 'sys': {type(e).__name__}: {e}")
```

<details>

<summary>
RuntimeError raised for built-in module 'sys'
</summary>
```
Error for 'sys': RuntimeError: No root path can be found for the provided module 'sys'. This can happen because the module came from an import hook that does not provide file name information or because it's a namespace package. In this case the root path needs to be explicitly provided.

```
</details>

## Why This Is A Bug

The function's docstring explicitly states: "Find the root path of a package, or the path that contains a module. **If it cannot be found, returns the current working directory.**" This establishes a clear contract that the function should never raise an exception when unable to determine a root path.

However, the implementation violates this contract in a specific scenario: when called with built-in modules like `sys`, `builtins`, or `marshal` that have no `__file__` attribute, the function raises a `RuntimeError` instead of returning `os.getcwd()`.

The bug reveals an inconsistency in the function's behavior:
- For non-existent modules (e.g., `'this_module_does_not_exist_1234567890'`): Returns the current working directory as documented
- For built-in modules without `__file__` attribute: Raises `RuntimeError`, violating the documented contract

This inconsistent behavior suggests the exception for built-in modules was not intentional, but rather an oversight in the implementation.

## Relevant Context

The function is located in `/flask/helpers.py` (lines 570-624) and is marked with `:meta private:`, indicating it's primarily for internal Flask use. However, the documented contract should still be honored.

The issue occurs at line 614 where the code raises a `RuntimeError` when `filepath is None` after attempting to import the module. The comment above this block mentions namespace packages, but doesn't account for built-in modules which also have no `__file__` attribute.

Testing confirms the inconsistency:
- `get_root_path('this_module_does_not_exist_1234567890')` returns `/home/npc/pbt/agentic-pbt/worker_/12` (current directory)
- `get_root_path('sys')` raises `RuntimeError`

Flask repository: https://github.com/pallets/flask
Relevant source: https://github.com/pallets/flask/blob/main/src/flask/helpers.py

## Proposed Fix

```diff
--- a/flask/helpers.py
+++ b/flask/helpers.py
@@ -610,14 +610,8 @@ def get_root_path(import_name: str) -> str:
         # If we don't have a file path it might be because it is a
-        # namespace package. In this case pick the root path from the
-        # first module that is contained in the package.
+        # namespace package or built-in module. Fall back to cwd as documented.
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
```