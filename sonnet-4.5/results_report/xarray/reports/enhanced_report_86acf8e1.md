# Bug Report: xarray.util._deprecate_positional_args ValueError on Excess Arguments

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with `ValueError` when a decorated function is called with more positional arguments than it can accept, instead of allowing Python to raise its standard `TypeError`.

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
            pass  # This is the expected behavior
        except ValueError:
            assert False, "Raised ValueError instead of TypeError"

if __name__ == "__main__":
    test_excess_positional_args_should_raise_typeerror()
```

<details>

<summary>
**Failing input**: `decorated(0, 0, 0)` where `func(a, *, b=0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 15, in test_excess_positional_args_should_raise_typeerror
    decorated(x, y, z)
    ~~~~~~~~~^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 23, in <module>
    test_excess_positional_args_should_raise_typeerror()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 6, in test_excess_positional_args_should_raise_typeerror
    def test_excess_positional_args_should_raise_typeerror(x, y, z):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 20, in test_excess_positional_args_should_raise_typeerror
    assert False, "Raised ValueError instead of TypeError"
           ^^^^^
AssertionError: Raised ValueError instead of TypeError
Falsifying example: test_excess_positional_args_should_raise_typeerror(
    # The test always failed when commented parts were varied together.
    x=0,  # or any other generated value
    y=0,  # or any other generated value
    z=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import warnings
from xarray.util.deprecation_helpers import _deprecate_positional_args

def func(x, *, y=0):
    return x + y

# Create decorated version
decorated = _deprecate_positional_args("v0.1.0")(func)

# Try calling with too many positional arguments
print("Calling decorated(1, 2, 3) on func(x, *, y=0)...")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = decorated(1, 2, 3)
        print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

print("\n" + "="*50 + "\n")

# Show what happens with undecorated function for comparison
print("For comparison, calling undecorated func(1, 2, 3)...")
try:
    result = func(1, 2, 3)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
```

<details>

<summary>
ValueError instead of TypeError when calling decorated function with excess arguments
</summary>
```
Calling decorated(1, 2, 3) on func(x, *, y=0)...
Exception type: ValueError
Exception message: zip() argument 2 is longer than argument 1

==================================================

For comparison, calling undecorated func(1, 2, 3)...
Exception type: TypeError
Exception message: func() takes 1 positional argument but 3 were given
```
</details>

## Why This Is A Bug

This violates Python's expected error handling behavior. When a function receives too many positional arguments, Python should raise a `TypeError` with a clear message like "func() takes X positional arguments but Y were given". However, the `_deprecate_positional_args` decorator intercepts this case and crashes with a cryptic `ValueError: zip() argument 2 is longer than argument 1`.

The decorator's documented purpose is to issue deprecation warnings for valid-but-deprecated usage of positional arguments that should become keyword-only. It should not alter Python's standard error handling for genuinely invalid function calls. The issue occurs in lines 112-115 of `deprecation_helpers.py` where `zip()` with `strict=True` tries to pair lists of different lengths when the number of excess positional arguments exceeds the number of keyword-only parameters.

## Relevant Context

The `_deprecate_positional_args` decorator was adapted from scikit-learn to help xarray transition APIs from positional to keyword-only arguments. The decorator is used internally but affects public-facing xarray APIs that employ it.

The bug occurs in this specific scenario:
- Function signature: `func(x, *, y=0)` (1 positional, 1 keyword-only parameter)
- Call: `decorated(1, 2, 3)` (3 positional arguments)
- `n_extra_args = 3 - 1 = 2` (2 extra positional arguments)
- `kwonly_args[:2]` yields `['y']` (only 1 element available)
- `args[-2:]` yields `(2, 3)` (2 elements)
- `zip(['y'], (2, 3), strict=True)` raises `ValueError` due to mismatched lengths

The decorator code is located at: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py`

## Proposed Fix

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -97,6 +97,11 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
         def inner(*args, **kwargs):
             name = func.__name__
             n_extra_args = len(args) - len(pos_or_kw_args)
+
+            # If more positional args than the function can handle, let Python raise TypeError
+            if n_extra_args > len(kwonly_args):
+                return func(*args, **kwargs)
+
             if n_extra_args > 0:
                 extra_args = ", ".join(kwonly_args[:n_extra_args])
```