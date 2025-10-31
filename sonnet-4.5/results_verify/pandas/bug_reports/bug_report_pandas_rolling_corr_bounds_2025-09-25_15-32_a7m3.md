# Bug Report: pandas.core.window Rolling Correlation Exceeds Valid Bounds

**Target**: `pandas.core.window.rolling.Rolling.corr`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The rolling correlation calculation can return values slightly outside the mathematically valid range of [-1, 1] due to floating-point precision errors. While the errors are small (around machine epsilon, ~2e-16), they violate the mathematical definition of correlation and could cause issues in code that assumes this invariant.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=30),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=30)
)
def test_rolling_corr_bounds(values1, values2):
    assume(len(values1) == len(values2))

    s1 = pd.Series(values1)
    s2 = pd.Series(values2)

    result = s1.rolling(3).corr(s2)

    for i, val in enumerate(result):
        if not np.isnan(val):
            assert -1 <= val <= 1, f"At index {i}: correlation {val} outside [-1, 1]"
```

**Failing input**: `values1=[0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]`, `values2=[0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

values1 = [0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]
values2 = values1

s1 = pd.Series(values1)
s2 = pd.Series(values2)

result = s1.rolling(3).corr(s2)

print("Rolling correlation:", result.values)
print(f"Correlation at index 4: {result.iloc[4]}")
print(f"Is > 1.0? {result.iloc[4] > 1.0}")
print(f"Difference from 1.0: {result.iloc[4] - 1.0}")
```

**Output:**
```
Rolling correlation: [nan nan nan 1.0 1.0000000000000002 1.0 0.999999999999997]
Correlation at index 4: 1.0000000000000002
Is > 1.0? True
Difference from 1.0: 2.220446049250313e-16
```

## Why This Is A Bug

Correlation coefficients are mathematically bounded to the range [-1, 1]. Values outside this range are impossible. While the violation is small (~2e-16, approximately machine epsilon), it still represents incorrect output that violates the mathematical definition.

This could cause issues in:
1. Code that performs strict bounds checking
2. Numerical algorithms that assume correlation ∈ [-1, 1]
3. Statistical tests that use correlation values

Note that NumPy's `corrcoef` correctly returns exactly 1.0 for the same data.

## Fix

The correlation result should be clamped to the valid range [-1, 1]:

```diff
--- a/pandas/core/window/rolling.py
+++ b/pandas/core/window/rolling.py
@@ -correlation_calculation_location
+        # Ensure correlation is within valid bounds
+        result = np.clip(result, -1.0, 1.0)
         return result
```

This is a simple fix that ensures numerical precision errors don't cause mathematically invalid results.