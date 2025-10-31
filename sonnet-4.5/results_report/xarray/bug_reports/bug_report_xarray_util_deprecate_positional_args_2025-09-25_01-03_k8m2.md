# Bug Report: xarray.util _deprecate_positional_args ValueError Crash

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a decorated function is called with more positional arguments than it accepts, `_deprecate_positional_args` crashes with `ValueError` instead of letting Python raise the appropriate `TypeError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import warnings
from xarray.util.deprecation_helpers import _deprecate_positional_args

@given(st.integers(), st.integers(), st.integers())
def test_excess_positional_args_should_raise_typeerror(x, y, z):
    def func(a, *, b=0):
        return a + b

    decorated = _deprecate_positional_args("v0.1.0")(func)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            decorated(x, y, z)
            assert False, "Should have raised an exception"
        except TypeError:
            pass
        except ValueError:
            assert False, "Raised ValueError instead of TypeError"
```

**Failing input**: `decorated(1, 2, 3)` where `func(a, *, b=0)`

## Reproducing the Bug

```python
import warnings
from xarray.util.deprecation_helpers import _deprecate_positional_args

def func(x, *, y=0):
    return x + y

decorated = _deprecate_positional_args("v0.1.0")(func)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    decorated(1, 2, 3)
```

Output:
```
ValueError: zip() argument 2 is longer than argument 1
```

## Why This Is A Bug

When calling a function with too many positional arguments, Python normally raises `TypeError: func() takes 1 positional argument but 3 were given`. However, the `_deprecate_positional_args` decorator intercepts this and crashes with `ValueError` due to mismatched list lengths in the `zip()` call.

The issue occurs at line 112-115 of `deprecation_helpers.py`:
```python
zip_args = zip(
    kwonly_args[:n_extra_args], args[-n_extra_args:], strict=True
)
kwargs.update(zip_args)
```

When `n_extra_args > len(kwonly_args)`, the two arguments to `zip()` have different lengths, causing `ValueError` with `strict=True`.

Example:
- `func(1, 2, 3)` where `func(x, *, y=0)`
- `kwonly_args = ['y']` (1 element)
- `n_extra_args = 2`
- `kwonly_args[:2] = ['y']` (1 element)
- `args[-2:] = (2, 3)` (2 elements)
- `zip(['y'], (2, 3), strict=True)` → ValueError

## Fix

Add validation to check if too many positional arguments are passed before attempting to zip:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -97,6 +97,11 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
         def inner(*args, **kwargs):
             name = func.__name__
             n_extra_args = len(args) - len(pos_or_kw_args)
+
+            # Check if too many positional arguments provided
+            if n_extra_args > len(kwonly_args):
+                # Let Python raise the normal TypeError
+                return func(*args, **kwargs)
+
             if n_extra_args > 0:
                 extra_args = ", ".join(kwonly_args[:n_extra_args])
```

This fix allows the original function to raise the standard `TypeError` when too many arguments are provided, which is the expected Python behavior.