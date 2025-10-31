# Bug Report: scipy.stats.percentileofscore Exceeds Documented Range

**Target**: `scipy.stats.percentileofscore`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `percentileofscore` function returns values slightly exceeding 100 due to floating-point arithmetic, violating its documented range of [0, 100].

## Property-Based Test

```python
import numpy as np
import scipy.stats as stats
from hypothesis import given, strategies as st

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.sampled_from(['rank', 'weak', 'strict', 'mean'])
)
def test_percentileofscore_all_kinds_bounded(data, kind):
    score = data[0]
    result = stats.percentileofscore(data, score, kind=kind)
    if not np.isnan(result):
        assert 0 <= result <= 100
```

**Failing input**: `data=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], kind='rank'`

## Reproducing the Bug

```python
import scipy.stats as stats

data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
score = 1.0

result = stats.percentileofscore(data, score, kind='rank')
print(f"Result: {result}")
print(f"Result > 100: {result > 100}")
```

Output:
```
Result: 100.00000000000001
Result > 100: True
```

## Why This Is A Bug

The function's docstring explicitly states:

```
Returns
-------
pcos : float
    Percentile-position of score (0-100) relative to `a`.
```

The documented range is [0, 100], but due to floating-point arithmetic in the formula `(left + right + plus1) * (50.0 / n)`, the result can exceed 100 by a small epsilon. This violates the API contract.

## Fix

```diff
--- a/scipy/stats/_stats_py.py
+++ b/scipy/stats/_stats_py.py
@@ -126,13 +126,13 @@ def percentileofscore(a, score, kind='rank', nan_policy='propagate'):
             left = count(a < score)
             right = count(a <= score)
             plus1 = left < right
-            perct = (left + right + plus1) * (50.0 / n)
+            perct = np.clip((left + right + plus1) * (50.0 / n), 0, 100)
         elif kind == 'strict':
-            perct = count(a < score) * (100.0 / n)
+            perct = np.clip(count(a < score) * (100.0 / n), 0, 100)
         elif kind == 'weak':
-            perct = count(a <= score) * (100.0 / n)
+            perct = np.clip(count(a <= score) * (100.0 / n), 0, 100)
         elif kind == 'mean':
             left = count(a < score)
             right = count(a <= score)
-            perct = (left + right) * (50.0 / n)
+            perct = np.clip((left + right) * (50.0 / n), 0, 100)
         else:
             raise ValueError(
```