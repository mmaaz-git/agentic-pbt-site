# Bug Report: xarray.util._deprecate_positional_args Crashes With Too Many Positional Arguments

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with a `ValueError` when a function receives more positional arguments than it has keyword-only parameters. Instead of gracefully handling the error or raising a proper `TypeError`, it crashes during the zip operation.

## Property-Based Test

```python
import pytest
from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v0.1.0")
def func_one_kwonly(a, *, b=2):
    return a + b


def test_too_many_positional_args():
    with pytest.raises(TypeError):
        func_one_kwonly(1, 2, 3)
```

**Failing input**: `func_one_kwonly(1, 2, 3)` - passing 3 positional arguments to a function that accepts 1 positional and 1 keyword-only argument

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v0.1.0")
def example_func(a, *, b=2):
    return a + b


example_func(1, 2, 3)
```

Expected: `TypeError` indicating too many positional arguments
Actual: `ValueError: zip() argument 2 is longer than argument 1`

## Why This Is A Bug

When a decorated function receives too many positional arguments, Python should raise a `TypeError` with a clear message about the incorrect number of arguments. Instead, the decorator crashes with an obscure `ValueError` during internal processing. This happens because:

1. The function receives 3 positional args: `(1, 2, 3)`
2. It expects 1 positional arg (`a`) and has 1 keyword-only arg (`b`)
3. `n_extra_args = 3 - 1 = 2`
4. `kwonly_args[:2]` gives `['b']` (only 1 element)
5. `args[-2:]` gives `(2, 3)` (2 elements)
6. `zip(..., strict=True)` fails because lists have different lengths

The decorator should either:
- Let Python's normal TypeError propagate (preferred)
- Validate that `n_extra_args <= len(kwonly_args)` and raise a proper error

## Fix

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -96,6 +96,11 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
         @wraps(func)
         def inner(*args, **kwargs):
             name = func.__name__
             n_extra_args = len(args) - len(pos_or_kw_args)
+            if n_extra_args > len(kwonly_args):
+                # Too many positional arguments - let Python raise TypeError
+                # by calling the function with the extra args
+                return func(*args, **kwargs)
+
             if n_extra_args > 0:
                 extra_args = ", ".join(kwonly_args[:n_extra_args])