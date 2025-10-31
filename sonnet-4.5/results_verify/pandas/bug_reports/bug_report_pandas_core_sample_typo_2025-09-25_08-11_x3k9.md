# Bug Report: pandas.core.sample Error Message Typo

**Target**: `pandas.core.sample.preprocess_weights`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Error message in `preprocess_weights` function contains a typo: "many" instead of "may" on line 70.

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
```

**Failing input**: Any negative weights array, e.g., `[-0.5]`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3]})
weights = pd.Series([-1, 0, 1])

try:
    df.sample(n=1, weights=weights)
except ValueError as e:
    print(e)
```

Output:
```
weight vector many not include negative values
```

## Why This Is A Bug

The error message contains a grammatical error. It should read "may not include" rather than "many not include". This is a documentation/contract bug affecting user experience.

## Fix

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