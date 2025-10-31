# Bug Report: xarray.util._deprecate_positional_args VAR_POSITIONAL Crash

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with a `ValueError` when applied to functions that have `*args` (VAR_POSITIONAL parameters).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import _deprecate_positional_args
import warnings

@given(st.integers(), st.integers(), st.integers())
def test_decorator_with_varargs(x, y, z):
    @_deprecate_positional_args("v0.1.0")
    def func(*args, **kwargs):
        return sum(args) + sum(kwargs.values())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = func(x, y, a=z)
```

**Failing input**: Any call with multiple positional arguments, e.g., `func(1, 2, a=3)`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import _deprecate_positional_args
import warnings

@_deprecate_positional_args("v0.1.0")
def func(*args, **kwargs):
    return sum(args) + sum(kwargs.values())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = func(1, 2, a=3)
```

Output:
```
ValueError: zip() argument 2 is longer than argument 1
```

## Why This Is A Bug

The `_deprecate_positional_args` decorator is designed to help deprecate positional arguments in function signatures. However, it crashes when applied to functions with `*args` because:

1. The decorator only looks for `POSITIONAL_OR_KEYWORD`, `POSITIONAL_ONLY`, and `KEYWORD_ONLY` parameter kinds
2. It doesn't handle `VAR_POSITIONAL` (*args) parameters
3. When `kwonly_args` is empty (no KEYWORD_ONLY params) but positional args are passed, the `zip()` call fails

This violates the principle that decorators should either work correctly or fail gracefully at decoration time, not crash at call time with confusing errors.

## Fix

The decorator should detect `*args` during decoration and either:
1. Raise a clear error at decoration time if the function has `*args`
2. Skip the deprecation logic for functions with `*args`

Recommended fix - detect and skip functions with VAR_POSITIONAL:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -82,6 +82,7 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
     def _decorator(func):
         signature = inspect.signature(func)

+        has_var_positional = False
         pos_or_kw_args = []
         kwonly_args = []
         for name, param in signature.parameters.items():
@@ -89,11 +90,17 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
                 pos_or_kw_args.append(name)
             elif param.kind == KEYWORD_ONLY:
                 kwonly_args.append(name)
+            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
+                has_var_positional = True
                 if param.default is EMPTY:
                     # IMHO `def f(a, *, b):` does not make sense -> disallow it
                     # if removing this constraint -> need to add these to kwargs as well
                     raise TypeError("Keyword-only param without default disallowed.")

+        # Skip deprecation logic if function has *args
+        if has_var_positional:
+            return func
+
         @wraps(func)
         def inner(*args, **kwargs):
             name = func.__name__
```