# Bug Report: FixedForwardWindowIndexer Step Zero Crash

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` crashes with `ZeroDivisionError` when `step=0` is passed, instead of raising an informative `ValueError` explaining that step must be positive.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=0, max_value=100),
)
def test_step_zero_raises_informative_error(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
        indexer.get_window_bounds(num_values=num_values, step=0)

    if isinstance(exc_info.value, ZeroDivisionError):
        pytest.fail("Should raise ValueError with informative message, not ZeroDivisionError")
```

**Failing input**: `num_values=1, window_size=0, step=0`

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=5)
start, end = indexer.get_window_bounds(num_values=10, step=0)
```

Output:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "pandas/core/indexers/objects.py", line 340, in get_window_bounds
    start = np.arange(0, num_values, step, dtype="int64")
ZeroDivisionError: division by zero
```

## Why This Is A Bug

1. **Unhelpful error message**: `ZeroDivisionError: division by zero` doesn't explain that `step` must be positive
2. **No input validation**: The function should validate that `step > 0` and raise `ValueError` with a clear message
3. **Poor user experience**: Users get a confusing error that doesn't help them understand what went wrong
4. **API contract violation**: The parameter should be validated at the API boundary

## Fix

Add validation for the step parameter:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -334,6 +334,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
         if step is None:
             step = 1
+        if step <= 0:
+            raise ValueError(f"step must be positive, got {step}")

         start = np.arange(0, num_values, step, dtype="int64")
         end = start + self.window_size
```