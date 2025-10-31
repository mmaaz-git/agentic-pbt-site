# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Crashes with Unhelpful Error on Zero Step

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `FixedForwardWindowIndexer.get_window_bounds()` is called with `step=0`, it crashes with an unhelpful `ZeroDivisionError` from numpy's internal implementation instead of validating the parameter and raising a descriptive `ValueError`.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st, example
from pandas.api.indexers import FixedForwardWindowIndexer


@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=0, max_value=100),
)
@example(num_values=1, window_size=0)  # The specific failing case
def test_step_zero_raises_informative_error(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
        indexer.get_window_bounds(num_values=num_values, step=0)

    if isinstance(exc_info.value, ZeroDivisionError):
        pytest.fail("Should raise ValueError with informative message, not ZeroDivisionError")
```

<details>

<summary>
**Failing input**: `num_values=1, window_size=0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/53
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_step_zero_raises_informative_error FAILED                  [100%]

=================================== FAILURES ===================================
___________________ test_step_zero_raises_informative_error ____________________

    @given(
>       num_values=st.integers(min_value=1, max_value=100),
                   ^^^
        window_size=st.integers(min_value=0, max_value=100),
    )

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

num_values = 1, window_size = 0

    @given(
        num_values=st.integers(min_value=1, max_value=100),
        window_size=st.integers(min_value=0, max_value=100),
    )
    @example(num_values=1, window_size=0)  # The specific failing case
    def test_step_zero_raises_informative_error(num_values, window_size):
        indexer = FixedForwardWindowIndexer(window_size=window_size)

        with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
            indexer.get_window_bounds(num_values=num_values, step=0)

        if isinstance(exc_info.value, ZeroDivisionError):
>           pytest.fail("Should raise ValueError with informative message, not ZeroDivisionError")
E           Failed: Should raise ValueError with informative message, not ZeroDivisionError
E           Falsifying explicit example: test_step_zero_raises_informative_error(
E               num_values=1,
E               window_size=0,
E           )

hypo.py:18: Failed
=========================== short test summary info ============================
FAILED hypo.py::test_step_zero_raises_informative_error - Failed: Should rais...
============================== 1 failed in 0.35s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

# Create an indexer with a window size of 5
indexer = FixedForwardWindowIndexer(window_size=5)

# Try to get window bounds with step=0 (this should crash)
try:
    start, end = indexer.get_window_bounds(num_values=10, step=0)
    print(f"Success: start={start}, end={end}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ZeroDivisionError: division by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/repo.py", line 8, in <module>
    start, end = indexer.get_window_bounds(num_values=10, step=0)
                 ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py", line 340, in get_window_bounds
    start = np.arange(0, num_values, step, dtype="int64")
ZeroDivisionError: division by zero
Error type: ZeroDivisionError
Error message: division by zero
```
</details>

## Why This Is A Bug

This violates expected API behavior in several ways:

1. **Unhelpful error message**: The `ZeroDivisionError: division by zero` message doesn't explain what the user did wrong. The error comes from numpy's internal `arange` implementation when step=0, not from pandas validation.

2. **Missing parameter validation**: The method already validates other parameters (e.g., it raises `ValueError` for invalid `center` and `closed` arguments), but fails to validate `step`. Since step=0 is semantically meaningless (you can't step through values with a step size of 0), this should be caught and reported clearly.

3. **Inconsistent with pandas error handling patterns**: Other pandas indexers and methods validate their parameters and provide descriptive error messages. This method defaults `step=None` to 1 (line 337-338), showing it understands step semantics, but doesn't validate non-None values.

4. **Documentation gap**: Neither the pandas documentation nor the method's docstring specifies that step must be positive. The parameter is simply described as "step passed from the top level rolling API" with no constraints mentioned.

## Relevant Context

The bug occurs in `/pandas/core/indexers/objects.py` at line 340 in the `FixedForwardWindowIndexer.get_window_bounds()` method. The issue is that when `step=0` is passed, it's directly forwarded to `numpy.arange()` without validation:

```python
if step is None:
    step = 1

start = np.arange(0, num_values, step, dtype="int64")  # Line 340 - crashes here
```

The numpy documentation for `arange` doesn't explicitly document that `step=0` causes a `ZeroDivisionError`, making this behavior surprising to users. The error propagates from numpy's C implementation where division by zero occurs when calculating the array length.

Related code: https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexers/objects.py#L297-L346

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -336,6 +336,9 @@ class FixedForwardWindowIndexer(BaseIndexer):
             )
         if step is None:
             step = 1
+
+        if step <= 0:
+            raise ValueError(f"step must be a positive integer, got {step}")

         start = np.arange(0, num_values, step, dtype="int64")
         end = start + self.window_size
```