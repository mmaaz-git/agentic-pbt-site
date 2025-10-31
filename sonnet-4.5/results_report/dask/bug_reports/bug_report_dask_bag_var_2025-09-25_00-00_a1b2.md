# Bug Report: dask.bag.Bag.var Division by Zero

**Target**: `dask.bag.Bag.var`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Bag.var()` method crashes with `ZeroDivisionError` when the number of elements equals the degrees of freedom parameter (`n == ddof`). This occurs because the variance aggregation function divides by `(n - ddof)` without checking if it equals zero.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.bag as db

@given(st.integers(min_value=1, max_value=10))
def test_variance_division_by_zero(n):
    data = [float(i) for i in range(n)]
    b = db.from_sequence(data, npartitions=1)

    b.var(ddof=n).compute()
```

**Failing input**: Any bag with `n` elements when calling `.var(ddof=n)`

## Reproducing the Bug

```python
import dask.bag as db

data = [5.0]
b = db.from_sequence(data, npartitions=1)
result = b.var(ddof=1).compute()
```

Expected: Should raise a more informative error like `ValueError("variance requires at least ddof + 1 elements")`

Actual: Raises `ZeroDivisionError: division by zero`

## Why This Is A Bug

1. The function crashes on valid inputs rather than providing a clear error message
2. Sample variance (`ddof=1`) is a common statistical operation, and users may legitimately have single-element partitions or datasets
3. The error message doesn't indicate the actual problem (insufficient data for the requested ddof)
4. Other statistical libraries (NumPy, pandas) handle this case more gracefully with warnings or NaN

## Fix

```diff
--- a/dask/bag/chunk.py
+++ b/dask/bag/chunk.py
@@ -31,5 +31,7 @@ def var_chunk(seq):

 def var_aggregate(x, ddof):
     squares, totals, counts = list(zip(*x))
     x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
+    if n <= ddof:
+        raise ValueError(f"variance requires at least {ddof + 1} elements when ddof={ddof}, but got {n}")
     result = (x2 / n) - (x / n) ** 2
     return result * n / (n - ddof)
```