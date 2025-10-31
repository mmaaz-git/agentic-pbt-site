# Bug Report: xarray.util.deprecation_helpers._deprecate_positional_args Crashes with Internal ValueError on Excess Arguments

**Target**: `xarray.util.deprecation_helpers._deprecate_positional_args`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_deprecate_positional_args` decorator crashes with an internal ValueError about zip length mismatch when a decorated function is called with more positional arguments than it can accept, instead of allowing Python to raise the appropriate TypeError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import _deprecate_positional_args


@given(st.integers(min_value=2, max_value=5))
def test_excess_positional_args_should_not_crash(n_extra):
    """Test that the decorator handles excess positional arguments gracefully.

    When a function is called with more positional arguments than it can accept,
    Python should raise a TypeError. The decorator should not interfere with this
    by raising its own ValueError about zip lengths.
    """
    @_deprecate_positional_args("v1.0.0")
    def func(a, *, b=1):
        return a + b

    # Create arguments: 1 valid positional + n_extra excess arguments
    args = tuple(range(1 + n_extra))

    try:
        result = func(*args)
        # If we get here, the function accepted the arguments (shouldn't happen)
        print(f"Unexpected success with {len(args)} args: {result}")
    except TypeError as e:
        # This is the expected behavior - Python raises TypeError for too many args
        print(f"Good: Got expected TypeError with {len(args)} args: {e}")
    except ValueError as e:
        if "zip()" in str(e):
            # This is the bug - internal ValueError about zip
            raise AssertionError(f"BUG: Decorator raised internal ValueError about zip with {len(args)} args: {e}")
        else:
            # Some other ValueError (unexpected)
            print(f"Unexpected ValueError with {len(args)} args: {e}")


if __name__ == "__main__":
    # Run the test with different numbers of excess arguments
    test_excess_positional_args_should_not_crash()
    print("\nTest completed!")
```

<details>

<summary>
**Failing input**: `test_excess_positional_args_should_not_crash(n_extra=2)`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/44/hypo.py:21: FutureWarning: Passing 'b' as positional argument(s) to func was deprecated in version v1.0.0 and will raise an error two releases later. Please pass them as keyword arguments.
  result = func(*args)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 21, in test_excess_positional_args_should_not_crash
    result = func(*args)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 38, in <module>
    test_excess_positional_args_should_not_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 6, in test_excess_positional_args_should_not_crash
    def test_excess_positional_args_should_not_crash(n_extra):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 30, in test_excess_positional_args_should_not_crash
    raise AssertionError(f"BUG: Decorator raised internal ValueError about zip with {len(args)} args: {e}")
AssertionError: BUG: Decorator raised internal ValueError about zip with 3 args: zip() argument 2 is longer than argument 1
Falsifying example: test_excess_positional_args_should_not_crash(
    n_extra=2,
)
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v1.0.0")
def example_func(a, *, b=1):
    """A simple function with one positional and one keyword-only argument."""
    return a + b


# Try to call with too many positional arguments (3 arguments when it only accepts 1 positional)
result = example_func(1, 2, 3)
print(f"Result: {result}")
```

<details>

<summary>
ValueError: zip() argument 2 is longer than argument 1
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/44/repo.py:11: FutureWarning: Passing 'b' as positional argument(s) to func was deprecated in version v1.0.0 and will raise an error two releases later. Please pass them as keyword arguments.
  result = example_func(1, 2, 3)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 11, in <module>
    result = example_func(1, 2, 3)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 115, in inner
    kwargs.update(zip_args)
    ~~~~~~~~~~~~~^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
```
</details>

## Why This Is A Bug

This bug violates fundamental Python behavior and decorator transparency principles:

1. **Incorrect Exception Type**: Without the decorator, Python would raise a `TypeError` with a clear message like "example_func() takes 1 positional argument but 3 were given". The decorator instead crashes with a confusing `ValueError: zip() argument 2 is longer than argument 1`.

2. **Internal Implementation Leak**: The error message exposes internal implementation details (zip operation) that are meaningless to users. Users should never see implementation-specific errors from a deprecation decorator.

3. **Decorator Contract Violation**: Decorators should be transparent except for their intended purpose. This decorator's purpose is to warn about deprecated positional arguments, not to replace Python's argument validation with its own error messages.

4. **Logic Error in Lines 112-115**: The bug occurs because:
   - `n_extra_args = len(args) - len(pos_or_kw_args)` correctly calculates excess arguments
   - When calling with `func(1, 2, 3)`: `n_extra_args = 3 - 1 = 2`
   - The decorator tries to zip `kwonly_args[:2]` (which is just `['b']`, length 1) with `args[-2:]` (which is `[2, 3]`, length 2)
   - Since Python 3.10+, `zip(..., strict=True)` raises ValueError when iterables have different lengths
   - This happens when `n_extra_args > len(kwonly_args)`

## Relevant Context

The `_deprecate_positional_args` decorator is adapted from scikit-learn and is used throughout xarray to deprecate positional arguments in favor of keyword-only arguments. It's meant to provide a smooth migration path by warning users before enforcing the change.

The decorator should only intervene when:
1. Extra positional arguments exist (`n_extra_args > 0`)
2. Those extra arguments can be mapped to keyword-only parameters

When more positional arguments are provided than the total number of parameters the function can accept (positional + keyword-only), the decorator should step aside and let Python's normal TypeError be raised.

Documentation: The decorator includes a docstring with examples but doesn't document the edge case behavior when excess arguments are provided.

Code location: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py:112-115`

## Proposed Fix

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -98,6 +98,11 @@ def _deprecate_positional_args(version) -> Callable[[T], T]:
             name = func.__name__
             n_extra_args = len(args) - len(pos_or_kw_args)
             if n_extra_args > 0:
+                # Check if too many arguments were passed
+                if n_extra_args > len(kwonly_args):
+                    # Let Python raise the appropriate TypeError
+                    return func(*args, **kwargs)
+
                 extra_args = ", ".join(kwonly_args[:n_extra_args])

                 warnings.warn(
```