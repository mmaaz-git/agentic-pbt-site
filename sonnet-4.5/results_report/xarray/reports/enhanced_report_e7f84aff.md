# Bug Report: xarray.util.deprecation_helpers._deprecate_positional_args Crashes With Excess Positional Arguments

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with a `ValueError` when a decorated function receives more positional arguments than the total number of parameters it can accept (positional + keyword-only).

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


if __name__ == "__main__":
    test_too_many_positional_args()
```

<details>

<summary>
**Failing input**: `func_one_kwonly(1, 2, 3)`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/0/hypo.py:12: FutureWarning: Passing 'b' as positional argument(s) to func_one_kwonly was deprecated in version v0.1.0 and will raise an error two releases later. Please pass them as keyword arguments.
  func_one_kwonly(1, 2, 3)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 16, in <module>
    test_too_many_positional_args()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 12, in test_too_many_positional_args
    func_one_kwonly(1, 2, 3)
    ~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v0.1.0")
def example_func(a, *, b=2):
    return a + b


# This should raise TypeError but crashes with ValueError instead
example_func(1, 2, 3)
```

<details>

<summary>
ValueError: zip() argument 2 is longer than argument 1
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/0/repo.py:10: FutureWarning: Passing 'b' as positional argument(s) to example_func was deprecated in version v0.1.0 and will raise an error two releases later. Please pass them as keyword arguments.
  example_func(1, 2, 3)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/repo.py", line 10, in <module>
    example_func(1, 2, 3)
    ~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
```
</details>

## Why This Is A Bug

This violates Python's expected behavior for function calls with too many positional arguments. In normal Python, calling a function with excess positional arguments produces a clear `TypeError` message:

```python
def func(a, *, b=2):
    return a + b

func(1, 2, 3)  # TypeError: func() takes 1 positional argument but 3 were given
```

However, the `_deprecate_positional_args` decorator intercepts the call and crashes during internal processing. The crash occurs because:

1. The decorated function `example_func(a, *, b=2)` receives 3 positional arguments: `(1, 2, 3)`
2. It has 1 positional parameter (`a`) and 1 keyword-only parameter (`b`)
3. The decorator calculates `n_extra_args = len(args) - len(pos_or_kw_args) = 3 - 1 = 2`
4. It attempts to create pairs with `zip(kwonly_args[:n_extra_args], args[-n_extra_args:], strict=True)`
5. But `kwonly_args[:2]` yields `['b']` (only 1 element) while `args[-2:]` yields `(2, 3)` (2 elements)
6. The `strict=True` flag causes `zip()` to raise `ValueError` when the iterables have different lengths

The decorator's purpose is to deprecate passing keyword-only arguments as positional arguments, but it should gracefully handle cases where there are genuinely too many arguments that exceed even the keyword-only parameters.

## Relevant Context

The `_deprecate_positional_args` decorator is adapted from scikit-learn and is designed to help library maintainers transition parameters from positional-or-keyword to keyword-only. The decorator intercepts calls where extra positional arguments are passed, issues a deprecation warning, and converts those positional arguments to keyword arguments.

The bug occurs specifically in lines 112-115 of `/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py`:

```python
zip_args = zip(
    kwonly_args[:n_extra_args], args[-n_extra_args:], strict=True
)
kwargs.update(zip_args)
```

The `strict=True` parameter was likely added to ensure the zip operation is consistent, but it doesn't account for the case where more positional arguments are provided than there are keyword-only parameters to convert them to.

## Proposed Fix

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -98,6 +98,11 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
         def inner(*args, **kwargs):
             name = func.__name__
             n_extra_args = len(args) - len(pos_or_kw_args)
+
+            # If there are more extra args than keyword-only params, let Python raise the TypeError
+            if n_extra_args > len(kwonly_args):
+                return func(*args, **kwargs)
+
             if n_extra_args > 0:
                 extra_args = ", ".join(kwonly_args[:n_extra_args])

@@ -110,9 +115,7 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
                     stacklevel=2,
                 )

-                zip_args = zip(
-                    kwonly_args[:n_extra_args], args[-n_extra_args:], strict=True
-                )
+                zip_args = zip(kwonly_args[:n_extra_args], args[-n_extra_args:])
                 kwargs.update(zip_args)

                 return func(*args[:-n_extra_args], **kwargs)
```