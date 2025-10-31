# Bug Report: pandas.core.window.rolling Rolling Window step=0 Validation Failure

**Target**: `pandas.core.window.rolling.BaseWindow._validate`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The validation for the `step` parameter in pandas rolling windows incorrectly accepts `step=0`, which causes a ZeroDivisionError during computation when numpy's arange function is called with step=0.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=5, max_size=20),
    st.integers(min_value=0, max_value=5)
)
def test_rolling_step_validation(data, step):
    """
    Property: Creating a rolling window with any step value should either
    succeed and allow aggregations, or fail during validation with a clear error.
    It should NOT pass validation and then crash during computation.
    """
    df = pd.DataFrame({'A': data})

    try:
        rolling = df.rolling(window=2, step=step)
        result = rolling.mean()
        assert result is not None
    except ValueError as e:
        # This should be raised during validation, not computation
        if "step must be" in str(e):
            # This is expected validation error - good!
            pass
        else:
            # If it's some other ValueError during computation, that's bad
            raise AssertionError(
                f"step={step} passed validation but crashed during computation with: {e}"
            )
    except ZeroDivisionError as e:
        # ZeroDivisionError should never happen - it means validation failed
        raise AssertionError(
            f"step={step} passed validation but crashed during computation with ZeroDivisionError: {e}"
        )

if __name__ == "__main__":
    # Run the test
    test_rolling_step_validation()
```

<details>

<summary>
**Failing input**: `step=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 20, in test_rolling_step_validation
    result = rolling.mean()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 2259, in mean
    return super().mean(
           ~~~~~~~~~~~~^
        numeric_only=numeric_only,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        engine=engine,
        ^^^^^^^^^^^^^^
        engine_kwargs=engine_kwargs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 1625, in mean
    return self._apply(window_func, name="mean", numeric_only=numeric_only)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 619, in _apply
    return self._apply_columnwise(homogeneous_func, name, numeric_only)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 494, in _apply_columnwise
    res = homogeneous_func(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 614, in homogeneous_func
    result = calc(values)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 602, in calc
    start, end = window_indexer.get_window_bounds(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        num_values=len(x),
        ^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        step=self.step,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py", line 110, in get_window_bounds
    end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
ZeroDivisionError: division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 40, in <module>
    test_rolling_step_validation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 6, in test_rolling_step_validation
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 34, in test_rolling_step_validation
    raise AssertionError(
        f"step={step} passed validation but crashed during computation with ZeroDivisionError: {e}"
    )
AssertionError: step=0 passed validation but crashed during computation with ZeroDivisionError: division by zero
Falsifying example: test_rolling_step_validation(
    data=[0.0, 0.0, 0.0, 0.0, 0.0],  # or any other generated value
    step=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/35/hypo.py:22
```
</details>

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]})
rolling = df.rolling(window=2, step=0)
result = rolling.mean()
print(result)
```

<details>

<summary>
ZeroDivisionError during mean() computation
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/repo.py", line 5, in <module>
    result = rolling.mean()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 2259, in mean
    return super().mean(
           ~~~~~~~~~~~~^
        numeric_only=numeric_only,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        engine=engine,
        ^^^^^^^^^^^^^^
        engine_kwargs=engine_kwargs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 1625, in mean
    return self._apply(window_func, name="mean", numeric_only=numeric_only)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 619, in _apply
    return self._apply_columnwise(homogeneous_func, name, numeric_only)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 494, in _apply_columnwise
    res = homogeneous_func(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 614, in homogeneous_func
    result = calc(values)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/window/rolling.py", line 602, in calc
    start, end = window_indexer.get_window_bounds(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        num_values=len(x),
        ^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        step=self.step,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py", line 110, in get_window_bounds
    end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
ZeroDivisionError: division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior because the validation in `BaseWindow._validate()` (pandas/core/window/rolling.py:212-213) only checks if `step < 0`, allowing `step=0` to pass through:

```python
if self.step < 0:
    raise ValueError("step must be >= 0")
```

However, `step=0` is both semantically invalid and technically problematic:

1. **Semantically invalid**: A step of 0 would mean the rolling window never advances, which makes no logical sense for a rolling operation.

2. **Technically problematic**: The step value is passed directly to `np.arange()` in `FixedWindowIndexer.get_window_bounds()` (pandas/core/indexers/objects.py:110), which raises a ZeroDivisionError when step=0.

3. **Documentation ambiguity**: The pandas documentation states that step is "equivalent to slicing as `[::step]`". In Python, attempting to slice with `[::0]` raises a ValueError ("slice step cannot be zero"), so by this analogy, step=0 should be invalid.

4. **Defensive programming exists but is incomplete**: The code already has defensive checks like `(self.step or 1)` in line 223 of rolling.py to avoid issues with step=0, indicating the developers were aware of potential problems with step=0, but the validation was not updated to reject it.

## Relevant Context

- **Pandas Version**: 2.3.2
- **Python Version**: 3.13
- **Error Location**: The ZeroDivisionError occurs in `/pandas/core/indexers/objects.py:110` when calling `np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")`
- **Validation Location**: The incomplete validation is in `/pandas/core/window/rolling.py:212-213`
- **Documentation Reference**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

The issue affects all rolling window operations (mean, sum, std, etc.) when step=0 is used, as they all go through the same `get_window_bounds()` code path.

## Proposed Fix

```diff
--- a/pandas/core/window/rolling.py
+++ b/pandas/core/window/rolling.py
@@ -209,8 +209,8 @@ class BaseWindow(SelectionMixin):
         if self.step is not None:
             if not is_integer(self.step):
                 raise ValueError("step must be an integer")
-            if self.step < 0:
-                raise ValueError("step must be >= 0")
+            if self.step <= 0:
+                raise ValueError("step must be > 0")

     def _check_window_bounds(
         self, start: np.ndarray, end: np.ndarray, num_vals: int
```