# Bug Report: Rolling variance returns negative values due to numerical precision issues

**Target**: `pandas.core.window.rolling.Rolling.var`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The rolling variance calculation can return small negative values due to numerical precision issues, violating the mathematical property that variance must always be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=5, max_size=20),
    st.integers(min_value=2, max_value=10),
)
def test_rolling_var_always_nonnegative(values, window):
    s = pd.Series(values)
    result = s.rolling(window=window).var()

    for val in result:
        if not pd.isna(val):
            assert val >= 0
```

**Failing input**: `values=[0.0, 48576.48503035901, 999999.5253488768, 0.0, 0.0078125]`, `window=2`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

values = [0.0, 48576.48503035901, 999999.5253488768, 0.0, 0.0078125]
s = pd.Series(values)

result = s.rolling(window=2).var()
print(result)
```

Output shows negative variance at index 4:
```
0         NaN
1    1.181098e+09
2    8.464881e+10
3    4.999999e+11
4   -3.051758e-05
dtype: float64
```

For the window `[0.0, 0.0078125]`, the variance should be approximately `3.05e-05` (positive), not `-3.05e-05`.

## Why This Is A Bug

Variance is mathematically defined as the average squared deviation from the mean and must always be non-negative. The negative value, while small in magnitude, violates this fundamental property and can cause issues:

1. **Square root fails**: `rolling.std()` relies on `sqrt(var)`, and `sqrt(negative)` returns NaN
2. **Boolean logic breaks**: Code like `if variance < 0` should never be true
3. **Statistical computations**: Downstream calculations may produce incorrect results

The issue appears to be a numerical precision problem in the rolling variance calculation. NumPy's `var()` function correctly returns a non-negative value for the same data:

```python
import numpy as np
np.var([0.0, 0.0078125], ddof=1)  # Returns 3.0517578125e-05 (positive)
```

## Fix

The variance calculation should clamp small negative values (likely due to floating-point arithmetic) to zero:

```diff
--- a/pandas/_libs/window/aggregations.pyx
+++ b/pandas/_libs/window/aggregations.pyx
@@ -XXX,X +XXX,X @@ cdef inline float64_t calc_var(XXX):
     # ... existing variance calculation ...
-    return result
+    # Clamp small negative values to zero to maintain var >= 0 invariant
+    return max(0.0, result)
```

Note: The exact location of the fix depends on the specific variance implementation used. The variance calculation may be in C/Cython code in `pandas/_libs/window/aggregations.pyx` or similar files. The key principle is that the final variance value should be clamped to ensure it's never negative.