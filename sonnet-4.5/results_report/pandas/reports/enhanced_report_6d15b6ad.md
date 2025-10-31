# Bug Report: pandas.core.sample.preprocess_weights Typo in Error Message

**Target**: `pandas.core.sample.preprocess_weights`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message for negative weights contains a grammatical typo: "many" instead of "may" in the ValueError message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
import numpy as np
from pandas import DataFrame
from pandas.core.sample import preprocess_weights

@given(
    n_rows=st.integers(min_value=1, max_value=100),
    axis=st.integers(min_value=0, max_value=1)
)
def test_preprocess_weights_negative_error_message(n_rows, axis):
    df = DataFrame(np.random.randn(n_rows, 3))
    shape = n_rows if axis == 0 else 3
    weights = np.ones(shape, dtype=np.float64)
    weights[0] = -1.0

    with pytest.raises(ValueError) as exc_info:
        preprocess_weights(df, weights, axis)

    error_msg = str(exc_info.value)
    # Bug: message says "many" instead of "may"
    assert "weight vector many not include negative values" == error_msg
```

<details>

<summary>
**Failing input**: `n_rows=1, axis=0`
</summary>
```
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=1,
    axis=0,
)
n_rows=1, axis=0: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=57,
    axis=0,
)
n_rows=57, axis=0: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=26,
    axis=1,
)
n_rows=26, axis=1: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=40,
    axis=1,
)
n_rows=40, axis=1: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=78,
    axis=1,
)
n_rows=78, axis=1: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=29,
    axis=0,
)
n_rows=29, axis=0: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=98,
    axis=1,
)
n_rows=98, axis=1: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=66,
    axis=0,
)
n_rows=66, axis=0: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=52,
    axis=0,
)
n_rows=52, axis=0: Error message confirmed - "weight vector many not include negative values"
Trying example: test_preprocess_weights_negative_error_message(
    n_rows=80,
    axis=1,
)
n_rows=80, axis=1: Error message confirmed - "weight vector many not include negative values"

All tests passed - error message consistently contains typo: "many" instead of "may"
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas import DataFrame
from pandas.core.sample import preprocess_weights

# Create a DataFrame with 5 rows and 3 columns
df = DataFrame(np.random.randn(5, 3))

# Create weights array with a negative value
weights = np.array([1.0, 2.0, -1.0, 3.0, 4.0])

# Try to preprocess weights with a negative value
try:
    preprocess_weights(df, weights, axis=0)
except ValueError as e:
    print(f"Error message: {e}")
```

<details>

<summary>
ValueError with typo: "weight vector many not include negative values"
</summary>
```
Error message: weight vector many not include negative values
```
</details>

## Why This Is A Bug

This violates expected behavior because the error message contains a grammatical error that makes it unprofessional and potentially confusing. The word "many" does not make grammatical sense in this context - the correct modal verb should be "may" to indicate prohibition. This is inconsistent with the error message for infinite values on line 67 of the same file, which correctly uses "may not include". The typo appears in pandas/core/sample.py:70 where it raises a ValueError for negative weights. While the functionality works correctly (negative weights are properly rejected), the error message text itself is incorrect and should be fixed for clarity and professionalism.

## Relevant Context

This bug is located in the `preprocess_weights` function which validates weights before they're used in DataFrame.sample() and related sampling operations. The function performs several validation checks:

1. Line 64: Validates that weight length matches the axis being sampled
2. Line 67: Checks for infinite values with message "weight vector may not include `inf` values" (correct grammar)
3. Line 70: Checks for negative values with message "weight vector many not include negative values" (typo - should be "may")
4. Lines 72-76: Handles NaN values by setting them to zero

The inconsistency between the infinite values error message (correct) and negative values error message (typo) makes this clearly unintentional. This function is part of the public API as it's called when users use DataFrame.sample() with weights.

Relevant pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html

Source code location: pandas/core/sample.py:70

## Proposed Fix

```diff
--- a/pandas/core/sample.py
+++ b/pandas/core/sample.py
@@ -67,7 +67,7 @@ def preprocess_weights(obj: NDFrame, weights, axis: AxisInt) -> np.ndarray:
         raise ValueError("weight vector may not include `inf` values")

     if (weights < 0).any():
-        raise ValueError("weight vector many not include negative values")
+        raise ValueError("weight vector may not include negative values")

     missing = np.isnan(weights)
     if missing.any():
```