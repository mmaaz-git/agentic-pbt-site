# Bug Report: pandas.core.window.common.prep_binary Infinity Converts Finite Values to NaN

**Target**: `pandas.core.window.common.prep_binary`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `prep_binary` function incorrectly converts finite values to NaN when aligning two Series if one contains infinity. This happens because the function uses `0 * inf` which mathematically equals NaN.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.window.common import prep_binary


@given(st.lists(st.floats(allow_nan=False, allow_infinity=True), min_size=1, max_size=50))
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

if __name__ == "__main__":
    test_prep_binary_infinity_handling()
```

<details>

<summary>
**Failing input**: `values=[inf]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 26, in <module>
    test_prep_binary_infinity_handling()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 8, in test_prep_binary_infinity_handling
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 22, in test_prep_binary_infinity_handling
    assert not np.isnan(X.iloc[i]), \
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Finite value s1[0]=1.0 became NaN due to inf in s2[0]
Falsifying example: test_prep_binary_infinity_handling(
    values=[inf],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/39/hypo.py:14
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/common.py:308
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/construction.py:661
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:138
```
</details>

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

<details>

<summary>
Output shows finite value becoming NaN
</summary>
```
s1: [1. 2. 3.]
s2: [10. inf 30.]
X: [ 1. nan  3.]
Y: [10. inf 30.]
```
</details>

## Why This Is A Bug

The `prep_binary` function is used throughout pandas' window operations (rolling correlations, covariances, etc.) to align two Series and propagate NaN values between them. The current implementation at lines 166-167 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/window/common.py`:

```python
X = arg1 + 0 * arg2
Y = arg2 + 0 * arg1
```

This clever trick aligns indices and was intended to propagate NaN values (since `anything * NaN = NaN`). However, it has an unintended side effect: `0 * inf = NaN` in floating-point arithmetic, so any infinity value in one Series causes the corresponding finite value in the other Series to become NaN.

This violates the expected behavior where:
1. NaN values should propagate between aligned Series (intended behavior)
2. Infinity values should remain as infinity (not convert to NaN)
3. Finite values should remain finite (not convert to NaN due to infinity in the other Series)

The bug silently corrupts data in statistical calculations, producing incorrect results without any warning when legitimate infinity values are present in financial or scientific data.

## Relevant Context

The `prep_binary` function is called by `flex_binary_moment` which is used for pairwise statistical operations in pandas' rolling window calculations. This affects methods like:
- `DataFrame.rolling().corr()`
- `DataFrame.rolling().cov()`
- Any other rolling window operation that compares two Series

The function's purpose is documented as "mask out values, this also makes a common index..." but the implementation has this mathematical edge case with infinity values.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.html

## Proposed Fix

Replace the multiplication-based approach with explicit index alignment and NaN propagation:

```diff
def prep_binary(arg1, arg2):
-    # mask out values, this also makes a common index...
-    X = arg1 + 0 * arg2
-    Y = arg2 + 0 * arg1
+    # Align indices and propagate NaN values without converting inf to nan
+    X, Y = arg1.align(arg2, join='outer')
+
+    # Propagate NaN values: if either series has NaN at a position, both should
+    nan_mask1 = X.isna()
+    nan_mask2 = Y.isna()
+
+    X = X.copy()
+    Y = Y.copy()
+    X[nan_mask2] = np.nan
+    Y[nan_mask1] = np.nan

    return X, Y
```