# Bug Report: pandas.core.window.common.prep_binary Infinity Converted to NaN

**Target**: `pandas.core.window.common.prep_binary`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `prep_binary` function silently converts infinity values to NaN when aligning two Series. When one Series contains infinity at index i and the other contains a finite value at index i, the finite value is incorrectly converted to NaN.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.window.common import prep_binary


@given(st.lists(st.floats(allow_nan=False, allow_infinity=True, min_value=-1e100, max_value=1e100), min_size=1, max_size=50))
@settings(max_examples=1000)
def test_prep_binary_infinity_handling(values):
    has_inf = any(np.isinf(v) for v in values)
    if not has_inf:
        return

    s1_values = [1.0] * len(values)
    s1 = pd.Series(s1_values)
    s2 = pd.Series(values)

    X, Y = prep_binary(s1, s2)

    for i in range(len(values)):
        if np.isinf(s2.iloc[i]) and not np.isnan(s1.iloc[i]):
            assert not np.isnan(X.iloc[i]), \
                f"Finite value s1[{i}]={s1.iloc[i]} became NaN due to inf in s2[{i}]"
```

**Failing input**: `s1 = [1.0, 2.0, 3.0]`, `s2 = [10.0, inf, 30.0]`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.window.common import prep_binary

s1 = pd.Series([1.0, 2.0, 3.0])
s2 = pd.Series([10.0, np.inf, 30.0])

X, Y = prep_binary(s1, s2)

print(f"s1: {s1.values}")
print(f"s2: {s2.values}")
print(f"X: {X.values}")
print(f"Y: {Y.values}")

assert not np.isnan(s1.iloc[1]), "s1[1] is finite"
assert np.isinf(s2.iloc[1]), "s2[1] is infinity"
assert np.isnan(X.iloc[1]), "BUG: X[1] became NaN!"
```

## Why This Is A Bug

The `prep_binary` function's purpose is to align indices and propagate NaN values between two Series for pairwise statistical operations. However, the implementation uses `s1 + 0 * s2`, and in NumPy/pandas, `0 * inf = nan`. This causes infinity values in one Series to silently convert finite values in the other Series to NaN, which is not the intended "NaN propagation" behavior.

This violates the principle of least surprise - users would expect that:
1. NaN values propagate between Series (intentional masking)
2. Infinity values remain as infinity
3. Finite values remain finite

Instead, infinity values are silently converted to NaN, affecting statistical calculations like rolling correlation and covariance.

## Fix

Replace the multiplication-based approach with an explicit NaN propagation:

```diff
def prep_binary(arg1, arg2):
-    # mask out values, this also makes a common index...
-    X = arg1 + 0 * arg2
-    Y = arg2 + 0 * arg1
+    # Align indices and propagate NaN values
+    from pandas import Series, DataFrame
+
+    if isinstance(arg1, Series):
+        X, Y = arg1.align(arg2, join='outer')
+        X = X.where(~Y.isna(), other=np.nan)
+        Y = Y.where(~X.align(arg2, join='outer')[0].isna(), other=np.nan)
+    else:
+        X, Y = arg1.align(arg2, join='outer')
+        X = X.where(~Y.isna(), other=np.nan)
+        Y = Y.where(~X.align(arg2, join='outer')[0].isna(), other=np.nan)

    return X, Y
```

Alternatively, a simpler fix that preserves the current implementation style but explicitly handles NaN:

```diff
def prep_binary(arg1, arg2):
-    # mask out values, this also makes a common index...
-    X = arg1 + 0 * arg2
-    Y = arg2 + 0 * arg1
+    # Align indices and propagate NaN values
+    # Use addition to align, but create zero-filled rather than 0*arg to avoid inf->nan conversion
+    import pandas as pd
+
+    if isinstance(arg1, pd.Series):
+        aligned1, aligned2 = arg1.align(arg2, join='outer')
+        mask1 = aligned1.isna()
+        mask2 = aligned2.isna()
+        X = aligned1.copy()
+        Y = aligned2.copy()
+        X[mask2] = np.nan
+        Y[mask1] = np.nan
+    else:
+        aligned1, aligned2 = arg1.align(arg2, join='outer')
+        mask1 = aligned1.isna()
+        mask2 = aligned2.isna()
+        X = aligned1.copy()
+        Y = aligned2.copy()
+        X[mask2] = np.nan
+        Y[mask1] = np.nan

    return X, Y
```