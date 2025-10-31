# Bug Report: dask.bag.Bag.var Division by Zero When n Equals ddof

**Target**: `dask.bag.Bag.var`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Bag.var()` method crashes with `ZeroDivisionError` when computing variance on a dataset where the number of elements equals the degrees of freedom parameter (n == ddof), instead of handling this edge case gracefully like NumPy and pandas do.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.bag as db
import dask

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

@given(st.integers(min_value=1, max_value=10))
def test_variance_division_by_zero(n):
    data = [float(i) for i in range(n)]
    b = db.from_sequence(data, npartitions=1)

    b.var(ddof=n).compute()

if __name__ == "__main__":
    test_variance_division_by_zero()
```

<details>

<summary>
**Failing input**: `n=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 16, in <module>
    test_variance_division_by_zero()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 9, in test_variance_division_by_zero
    def test_variance_division_by_zero(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 13, in test_variance_division_by_zero
    b.var(ddof=n).compute()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    results = schedule(expr, keys, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py", line 2519, in empty_safe_aggregate
    return empty_safe_apply(func, parts2, is_last)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py", line 2510, in empty_safe_apply
    return func(part)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/chunk.py", line 36, in var_aggregate
    return result * n / (n - ddof)
           ~~~~~~~~~~~^~~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_variance_division_by_zero(
    n=1,
)
```
</details>

## Reproducing the Bug

```python
import dask.bag as db
import dask

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

data = [5.0]
b = db.from_sequence(data, npartitions=1)
result = b.var(ddof=1).compute()
print(f"Result: {result}")
```

<details>

<summary>
ZeroDivisionError at line 36 in var_aggregate
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 9, in <module>
    result = b.var(ddof=1).compute()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    results = schedule(expr, keys, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py", line 2519, in empty_safe_aggregate
    return empty_safe_apply(func, parts2, is_last)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py", line 2510, in empty_safe_apply
    return func(part)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/chunk.py", line 36, in var_aggregate
    return result * n / (n - ddof)
           ~~~~~~~~~~~^~~~~~~~~~~~
ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Mathematical Edge Case Not Handled**: When n == ddof, the variance formula divides by (n - ddof) = 0, which is mathematically undefined. The implementation in `/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/chunk.py:36` performs this division without any guards, causing a crash.

2. **Inconsistent with NumPy and pandas**: Both NumPy and pandas handle this edge case gracefully:
   - `np.var([5.0], ddof=1)` returns `nan` with a RuntimeWarning about "Degrees of freedom <= 0"
   - `pd.Series([5.0]).var(ddof=1)` returns `nan` without warnings
   - Dask crashes with `ZeroDivisionError`, making it incompatible as a drop-in replacement

3. **Poor Error Messaging**: The error "float division by zero" doesn't communicate the actual problem to users - that there are insufficient data points for the requested degrees of freedom. Users need to understand the stack trace to diagnose the issue.

4. **Common Use Case Affected**: Sample variance (ddof=1) is the standard unbiased estimator used throughout statistics. Single-element datasets or partitions can occur in distributed computing when data is filtered, grouped, or partitioned unevenly.

5. **Documentation Gap**: The dask.bag.Bag.var documentation (line 1140 in core.py) simply states "Variance" without specifying behavior for edge cases or minimum data requirements, leaving users to discover this limitation through crashes.

## Relevant Context

The variance calculation in dask is split into two phases:
- `var_chunk` (lines 23-29 in chunk.py): Computes local statistics (squares, totals, counts) for each partition
- `var_aggregate` (lines 32-36 in chunk.py): Combines statistics from all partitions and computes final variance

The bug occurs in the aggregation phase at line 36:
```python
return result * n / (n - ddof)  # Division by zero when n == ddof
```

The Bag.var method is defined at line 1139-1143 in `/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py` and uses the reduction pattern with these chunk and aggregate functions.

Related documentation:
- NumPy variance: https://numpy.org/doc/stable/reference/generated/numpy.var.html
- pandas Series.var: https://pandas.pydata.org/docs/reference/api/pandas.Series.var.html
- Statistical background: When ddof=1, we compute sample variance; when ddof=0, population variance

## Proposed Fix

```diff
--- a/dask/bag/chunk.py
+++ b/dask/bag/chunk.py
@@ -31,6 +31,11 @@ def var_chunk(seq):

 def var_aggregate(x, ddof):
     squares, totals, counts = list(zip(*x))
     x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
+
+    if n <= ddof:
+        import warnings
+        warnings.warn(f"Degrees of freedom <= 0 for variance (n={n}, ddof={ddof})")
+        return float('nan')
+
     result = (x2 / n) - (x / n) ** 2
     return result * n / (n - ddof)
```