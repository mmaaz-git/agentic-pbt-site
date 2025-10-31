# Bug Report: dask.bag.chunk var_aggregate Numerical Instability

**Target**: `dask.bag.chunk.var_aggregate` and `dask.bag.chunk.var_chunk`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The variance calculation in `dask.bag.chunk.var_aggregate` uses a numerically unstable formula that produces negative variance values and incorrect results for datasets with large values or small variance.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math
from dask.bag.chunk import var_chunk, var_aggregate

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=2, max_size=100))
def test_var_chunk_aggregate(values):
    chunk_result = var_chunk(values)
    aggregate_result = var_aggregate([chunk_result], ddof=0)

    mean = sum(values) / len(values)
    expected_variance = sum((x - mean) ** 2 for x in values) / len(values)

    assert aggregate_result >= 0, "Variance cannot be negative"
    assert math.isclose(aggregate_result, expected_variance,
                       rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `[-356335.16553451226, -356335.16553451226, -356335.16553451226]`

## Reproducing the Bug

```python
from dask.bag.chunk import var_chunk, var_aggregate

values = [-356335.16553451226, -356335.16553451226, -356335.16553451226]
chunk_result = var_chunk(values)
variance = var_aggregate([chunk_result], ddof=0)

print(f"Computed variance: {variance}")
print(f"Expected variance: 0.0")

values2 = [259284.59765625, 259284.59765625, 259284.59765625]
chunk_result2 = var_chunk(values2)
variance2 = var_aggregate([chunk_result2], ddof=0)

print(f"\nComputed variance: {variance2}")
print(f"Expected variance: 0.0")
```

Output:
```
Computed variance: -3.0517578125e-05
Expected variance: 0.0

Computed variance: 7.62939453125e-06
Expected variance: 0.0
```

## Why This Is A Bug

1. **Negative variance is mathematically impossible** - variance is defined as the average of squared deviations and cannot be negative
2. **Identical values must have zero variance** - when all values are the same, variance should be exactly 0
3. **Incorrect results for large values** - the algorithm produces significant relative errors

The root cause is the use of the computationally convenient but numerically unstable formula:

```
Var(X) = E[X²] - E[X]²
```

This formula suffers from catastrophic cancellation when the two terms are nearly equal (large values or small variance).

## Fix

```diff
--- a/dask/bag/chunk.py
+++ b/dask/bag/chunk.py
@@ -20,18 +20,30 @@ def groupby_tasks_group_hash(x, hash, grouper):
     return hash(grouper(x)), x


-def var_chunk(seq):
-    squares, total, n = 0.0, 0.0, 0
-    for x in seq:
-        squares += x**2
-        total += x
-        n += 1
-    return squares, total, n
+def var_chunk(seq):
+    n = 0
+    mean = 0.0
+    m2 = 0.0
+
+    for x in seq:
+        n += 1
+        delta = x - mean
+        mean += delta / n
+        delta2 = x - mean
+        m2 += delta * delta2
+
+    return m2, mean, n


 def var_aggregate(x, ddof):
-    squares, totals, counts = list(zip(*x))
-    x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
-    result = (x2 / n) - (x / n) ** 2
-    return result * n / (n - ddof)
+    m2_values, means, counts = list(zip(*x))
+
+    total_count = sum(counts)
+    if total_count == 0:
+        return 0.0
+
+    combined_mean = sum(m * c for m, c in zip(means, counts)) / total_count
+    combined_m2 = sum(m2 + c * (m - combined_mean) ** 2
+                     for m2, m, c in zip(m2_values, means, counts))
+
+    return combined_m2 / (total_count - ddof)
```

This uses Welford's online algorithm, which is numerically stable and avoids the catastrophic cancellation problem.