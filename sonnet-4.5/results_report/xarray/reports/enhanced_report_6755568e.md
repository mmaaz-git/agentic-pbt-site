# Bug Report: xarray.util.deprecation_helpers._deprecate_positional_args VAR_POSITIONAL Crash

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with a `ValueError` when applied to functions that have `*args` (VAR_POSITIONAL parameters) in their signature.

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

test_decorator_with_varargs()
```

<details>

<summary>
**Failing input**: `test_decorator_with_varargs(x=0, y=0, z=0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 15, in <module>
    test_decorator_with_varargs()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_decorator_with_varargs
    def test_decorator_with_varargs(x, y, z):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 13, in test_decorator_with_varargs
    result = func(x, y, a=z)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
Falsifying example: test_decorator_with_varargs(
    # The test always failed when commented parts were varied together.
    x=0,  # or any other generated value
    y=0,  # or any other generated value
    z=0,  # or any other generated value
)
```
</details>

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
    print(f"Result: {result}")
```

<details>

<summary>
ValueError: zip() argument 2 is longer than argument 1
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/repo.py", line 10, in <module>
    result = func(1, 2, a=3)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
```
</details>

## Why This Is A Bug

The `_deprecate_positional_args` decorator is designed to help deprecate positional arguments in function signatures, encouraging users to pass them as keyword arguments instead. However, the current implementation fails catastrophically when applied to functions with `*args` (VAR_POSITIONAL parameters).

The bug occurs because:

1. The decorator analyzes the function signature and only collects parameters of types `POSITIONAL_OR_KEYWORD`, `POSITIONAL_ONLY`, and `KEYWORD_ONLY` (lines 87-94 in deprecation_helpers.py)
2. It completely ignores `VAR_POSITIONAL` (*args) parameters
3. When a function with `*args` is called with positional arguments, `pos_or_kw_args` is empty (since *args is not counted)
4. The code calculates `n_extra_args = len(args) - len(pos_or_kw_args)` which becomes the total number of args passed
5. It then tries to zip `kwonly_args[:n_extra_args]` with `args[-n_extra_args:]`, but when there are no keyword-only arguments, this creates a zip with mismatched lengths
6. The `strict=True` parameter in the zip call (line 113) causes the ValueError

This violates the principle that decorators should either work correctly or fail gracefully at decoration time with a clear error message. Instead, this decorator silently accepts functions it cannot handle and crashes at runtime with a confusing error message that doesn't indicate the actual problem.

## Relevant Context

The decorator is adapted from scikit-learn's implementation as noted in the comments (lines 77-78). The original scikit-learn version may have similar limitations. The decorator's purpose is to facilitate API migrations by warning users when they use positional arguments that should be keyword-only.

The documentation example in the docstring (lines 63-74) shows the intended use case: converting a function like `def func(a, b=1):` to `def func(a, *, b=2):` to make `b` keyword-only. This works well for regular function signatures but doesn't account for variadic functions.

## Proposed Fix

The decorator should detect functions with `*args` during decoration and either skip the deprecation logic or raise a clear error at decoration time:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -81,14 +81,23 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:

     def _decorator(func):
         signature = inspect.signature(func)

+        has_var_positional = False
         pos_or_kw_args = []
         kwonly_args = []
         for name, param in signature.parameters.items():
             if param.kind in (POSITIONAL_OR_KEYWORD, POSITIONAL_ONLY):
                 pos_or_kw_args.append(name)
             elif param.kind == KEYWORD_ONLY:
                 kwonly_args.append(name)
                 if param.default is EMPTY:
                     # IMHO `def f(a, *, b):` does not make sense -> disallow it
                     # if removing this constraint -> need to add these to kwargs as well
                     raise TypeError("Keyword-only param without default disallowed.")
+            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
+                has_var_positional = True
+
+        # Skip deprecation logic if function has *args
+        if has_var_positional:
+            return func

         @wraps(func)
         def inner(*args, **kwargs):
```