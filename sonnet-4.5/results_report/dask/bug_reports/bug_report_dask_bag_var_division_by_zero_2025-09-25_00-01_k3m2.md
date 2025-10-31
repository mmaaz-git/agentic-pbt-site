# Bug Report: dask.bag.Bag.var Division by Zero

**Target**: `dask.bag.Bag.var`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Bag.var()` method crashes with `ZeroDivisionError` when `ddof == n` or when called on an empty bag.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=10))
def test_variance_no_crash(data):
    b = db.from_sequence(data if data else [0], npartitions=1)
    for ddof in range(0, len(data) + 2):
        try:
            var = b.var(ddof=ddof).compute()
        except ZeroDivisionError:
            assert False, f"var() should not crash with ZeroDivisionError for ddof={ddof}, n={len(data)}"
```

**Failing input**: `[1.0, 2.0]` with `ddof=2`, or `[]` with `ddof=0`

## Reproducing the Bug

```python
import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

b = db.from_sequence([1.0, 2.0], npartitions=1)
result = b.var(ddof=2).compute()
```

Output:
```
ZeroDivisionError: float division by zero
```

Also crashes on empty sequence:
```python
b = db.from_sequence([], npartitions=1)
result = b.var(ddof=0).compute()
```

## Why This Is A Bug

The variance calculation divides by `(n - ddof)` without checking if this denominator is zero. When `ddof == n`, or when the bag is empty (n=0), this causes a crash. The function should either raise a more informative error or handle these edge cases gracefully.

## Fix

The same fix as the negative variance bug handles both issues:

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