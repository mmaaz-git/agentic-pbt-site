# Bug Report: pandas.core.window.rolling step=0 Validation Bug

**Target**: `pandas.core.window.rolling.BaseWindow._validate`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The validation for the `step` parameter in rolling windows accepts `step=0`, but using `step=0` causes a crash during computation when slicing operations are performed. The validation should reject `step=0` to prevent this crash.

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
        if "slice step cannot be zero" in str(e):
            raise AssertionError(
                f"step={step} passed validation but crashed during computation"
            )
```

**Failing input**: `step=0`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]})
rolling = df.rolling(window=2, step=0)
result = rolling.mean()
```

Output:
```
ValueError: slice step cannot be zero
```

## Why This Is A Bug

The validation in `BaseWindow._validate()` checks that `step >= 0`:

```python
if self.step < 0:
    raise ValueError("step must be >= 0")
```

However, `step=0` is semantically invalid for a rolling window (it would mean never advancing the window), and it causes a crash during computation when the code attempts to slice with `step=0`:

```python
# Line 237 in rolling.py
else index[:: self.step]
```

Python raises `ValueError: slice step cannot be zero` when attempting to slice with step=0. This error should be caught during validation, not during computation.

Additionally, the code uses `(self.step or 1)` in some places (line 223) to avoid division by zero, which suggests the developers were aware that `step=0` is problematic, but the validation was not updated accordingly.

## Fix

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