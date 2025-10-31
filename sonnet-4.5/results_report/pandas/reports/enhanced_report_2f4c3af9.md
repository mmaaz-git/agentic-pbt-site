# Bug Report: pandas.core.sample.preprocess_weights Error Message Typo

**Target**: `pandas.core.sample.preprocess_weights`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message in the `preprocess_weights` function contains a grammatical typo: "many" instead of "may" on line 70 when rejecting negative weight values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import numpy as np

@given(st.lists(st.floats(min_value=-1.0, max_value=-0.01), min_size=1, max_size=10))
def test_negative_weights_error_message(negative_weights):
    """Error message should say 'may not' not 'many not'"""
    df = pd.DataFrame({'A': range(len(negative_weights))})
    weights = pd.Series(negative_weights)

    try:
        df.sample(n=1, weights=weights)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert "may not" in error_msg or "many not" in error_msg
        if "many not" in error_msg:
            raise AssertionError(f"Typo in error message: {error_msg}")

# Run the test
test_negative_weights_error_message()
```

<details>

<summary>
**Failing input**: `negative_weights=[-1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 12, in test_negative_weights_error_message
    df.sample(n=1, weights=weights)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 6135, in sample
    weights = sample.preprocess_weights(self, weights, axis)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/sample.py", line 70, in preprocess_weights
    raise ValueError("weight vector many not include negative values")
ValueError: weight vector many not include negative values

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 21, in <module>
    test_negative_weights_error_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 6, in test_negative_weights_error_message
    def test_negative_weights_error_message(negative_weights):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 18, in test_negative_weights_error_message
    raise AssertionError(f"Typo in error message: {error_msg}")
AssertionError: Typo in error message: weight vector many not include negative values
Falsifying example: test_negative_weights_error_message(
    negative_weights=[-1.0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

# Create a simple DataFrame
df = pd.DataFrame({'A': [1, 2, 3]})

# Create weights with a negative value
weights = pd.Series([-1, 0, 1])

# Try to sample with negative weights (this should raise an error)
try:
    result = df.sample(n=1, weights=weights)
except ValueError as e:
    print(f"ValueError: {e}")
```

<details>

<summary>
ValueError with incorrect grammar in error message
</summary>
```
ValueError: weight vector many not include negative values
```
</details>

## Why This Is A Bug

This violates expected behavior because the error message contains incorrect grammar. The message "weight vector many not include negative values" should read "weight vector may not include negative values".

The typo is inconsistent with the parallel error message on line 67 of the same file, which correctly uses "may not" when checking for infinite values: "weight vector may not include \`inf\` values". Both error messages follow the same pattern and should use consistent, grammatically correct language.

While the functionality of rejecting negative weights works correctly, the grammatical error in the error message affects the professional quality and clarity of the library's user-facing output.

## Relevant Context

The `preprocess_weights` function in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/sample.py` validates weight vectors for DataFrame and Series sampling operations. The function performs several validation checks:

1. Line 64: Validates that weights length matches the axis being sampled
2. Line 67: Rejects infinite values with message: "weight vector may not include \`inf\` values" (correct grammar)
3. Line 70: Rejects negative values with message: "weight vector many not include negative values" (typo: "many" instead of "may")
4. Lines 72-76: Handles NaN values by setting them to zero

The consistent pattern across error messages indicates that "may not" is the intended phrasing, making this a clear typo rather than intentional wording.

Documentation reference: [pandas.DataFrame.sample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html)

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