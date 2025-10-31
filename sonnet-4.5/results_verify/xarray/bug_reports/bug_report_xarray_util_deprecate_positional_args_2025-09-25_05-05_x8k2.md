# Bug Report: xarray.util _deprecate_positional_args Crash on Excess Arguments

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with a ValueError when a decorated function is called with more positional arguments than it can accept.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import _deprecate_positional_args


@st.composite
def func_and_args(draw):
    n_kwonly = draw(st.integers(min_value=1, max_value=3))

    @_deprecate_positional_args("v1.0.0")
    def test_func(a, **kwargs):
        return a

    test_func.__code__ = test_func.__code__.replace(
        co_kwonlyargcount=n_kwonly
    )

    n_excess = draw(st.integers(min_value=1, max_value=5))
    args = draw(st.lists(st.integers(), min_size=2+n_excess, max_size=2+n_excess))

    return test_func, tuple(args)


@given(st.integers(min_value=2, max_value=5))
def test_excess_positional_args_should_not_crash(n_extra):
    @_deprecate_positional_args("v1.0.0")
    def func(a, *, b=1):
        return a + b

    args = tuple(range(1 + n_extra))

    try:
        func(*args)
    except TypeError:
        pass
    except ValueError as e:
        if "zip()" in str(e):
            raise AssertionError(f"Decorator should not raise ValueError about zip: {e}")
```

**Failing input**: Calling `func(a, *, b=1)` with 3+ positional arguments

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v1.0.0")
def example_func(a, *, b=1):
    return a + b


example_func(1, 2, 3)
```

Output:
```
FutureWarning: Passing 'b' as positional argument(s) to example_func was deprecated...
Traceback (most recent call last):
  ...
ValueError: zip() argument 2 is longer than argument 1
```

## Why This Is A Bug

When a function decorated with `_deprecate_positional_args` is called with too many positional arguments (more than pos_or_kw + kwonly params), the decorator crashes with an internal ValueError about zip lengths instead of letting Python raise the appropriate TypeError about too many arguments.

The issue occurs at line 112-115 in deprecation_helpers.py:
- `n_extra_args = len(args) - len(pos_or_kw_args)` calculates excess arguments
- When `n_extra_args > len(kwonly_args)`, the zip fails because:
  - `kwonly_args[:n_extra_args]` has length `len(kwonly_args)` (smaller)
  - `args[-n_extra_args:]` has length `n_extra_args` (larger)

This violates user expectations - Python should raise a TypeError for too many arguments, not a ValueError about zip.

## Fix

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -97,7 +97,14 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
         def inner(*args, **kwargs):
             name = func.__name__
             n_extra_args = len(args) - len(pos_or_kw_args)
             if n_extra_args > 0:
+                # Check if too many arguments were passed
+                if n_extra_args > len(kwonly_args):
+                    # Let the original function raise TypeError
+                    return func(*args, **kwargs)
+
                 extra_args = ", ".join(kwonly_args[:n_extra_args])

                 warnings.warn(
```