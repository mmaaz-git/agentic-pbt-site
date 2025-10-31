# Bug Report: dask.bag.Bag.var Negative Variance

**Target**: `dask.bag.Bag.var`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Bag.var()` method returns negative variance when `ddof > n`, which violates the mathematical property that variance must always be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=10))
def test_variance_non_negative(data):
    b = db.from_sequence(data, npartitions=1)
    for ddof in range(0, len(data) + 2):
        var = b.var(ddof=ddof).compute()
        assert var >= 0, f"Variance must be non-negative, got {var} with ddof={ddof}, n={len(data)}"
```

**Failing input**: `[1.0, 2.0]` with `ddof=3`

## Reproducing the Bug

```python
import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

b = db.from_sequence([1.0, 2.0], npartitions=1)
result = b.var(ddof=3).compute()
print(f"Variance: {result}")
```

Output:
```
Variance: -0.5
```

## Why This Is A Bug

Variance is defined as the expected value of the squared deviation from the mean, which is always non-negative by definition. The code in `dask/bag/chunk.py:var_aggregate` computes `result * n / (n - ddof)`, but when `ddof > n`, this produces a negative result. This is a silent data corruption bug that could lead to incorrect statistical analyses.

## Fix

```diff
--- a/dask/bag/chunk.py
+++ b/dask/bag/chunk.py
@@ -31,6 +31,8 @@ def var_chunk(seq):

 def var_aggregate(x, ddof):
     squares, totals, counts = list(zip(*x))
     x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
+    if n <= ddof:
+        raise ValueError(f"ddof must be less than the number of elements (ddof={ddof}, n={n})")
     result = (x2 / n) - (x / n) ** 2
     return result * n / (n - ddof)
```