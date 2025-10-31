# Bug Report: dask.bag.Bag.var Returns Negative Variance When ddof > n

**Target**: `dask.bag.Bag.var`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Bag.var()` method returns negative variance values when `ddof > n` and throws `ZeroDivisionError` when `ddof = n`, violating the mathematical property that variance must always be non-negative.

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

# Run the test
if __name__ == "__main__":
    test_variance_non_negative()
```

<details>

<summary>
**Failing input**: `[0.0]` with `ddof=1` and `[100.0, 99.99999999999999]` with `ddof=0`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 16, in <module>
  |     test_variance_non_negative()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 8, in test_variance_non_negative
  |     def test_variance_non_negative(data):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 12, in test_variance_non_negative
    |     assert var >= 0, f"Variance must be non-negative, got {var} with ddof={ddof}, n={len(data)}"
    |            ^^^^^^^^
    | AssertionError: Variance must be non-negative, got -1.8189894035458565e-12 with ddof=0, n=2
    | Falsifying example: test_variance_non_negative(
    |     data=[100.0, 99.99999999999999],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 11, in test_variance_non_negative
    |     var = b.var(ddof=ddof).compute()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    |     (result,) = compute(self, traverse=False, **kwargs)
    |                 ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    |     results = schedule(expr, keys, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py", line 2519, in empty_safe_aggregate
    |     return empty_safe_apply(func, parts2, is_last)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py", line 2510, in empty_safe_apply
    |     return func(part)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/chunk.py", line 36, in var_aggregate
    |     return result * n / (n - ddof)
    |            ~~~~~~~~~~~^~~~~~~~~~~~
    | ZeroDivisionError: float division by zero
    | Falsifying example: test_variance_non_negative(
    |     data=[0.0],  # or any other generated value
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

b = db.from_sequence([1.0, 2.0], npartitions=1)
result = b.var(ddof=3).compute()
print(f"Variance: {result}")
print(f"Expected: Variance should be non-negative, but got {result}")
```

<details>

<summary>
Output shows negative variance (-0.5)
</summary>
```
Variance: -0.5
Expected: Variance should be non-negative, but got -0.5
```
</details>

## Why This Is A Bug

This violates fundamental mathematical properties and differs from established statistical libraries:

1. **Mathematical Violation**: Variance is defined as E[(X - μ)²], which is the expected value of squared deviations. Since squares of real numbers are always non-negative, variance must be ≥ 0 by definition.

2. **Inconsistent with NumPy/pandas**:
   - NumPy returns `inf` when ddof ≥ n (never negative)
   - pandas returns `nan` when ddof ≥ n (never negative)
   - Dask returns negative values when ddof > n and raises ZeroDivisionError when ddof = n

3. **Silent Data Corruption**: The function returns mathematically impossible results without warning, potentially corrupting downstream statistical analyses.

4. **Numerical Instability**: The test also revealed floating-point precision issues where variance can be slightly negative even with valid ddof=0 due to numerical errors in the computation formula.

## Relevant Context

The bug occurs in `/dask/bag/chunk.py:var_aggregate` at line 36:
```python
def var_aggregate(x, ddof):
    squares, totals, counts = list(zip(*x))
    x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
    result = (x2 / n) - (x / n) ** 2
    return result * n / (n - ddof)  # Line 36: Division by (n - ddof)
```

When `ddof > n`, the denominator `(n - ddof)` becomes negative, causing the positive variance to become negative. When `ddof = n`, it causes division by zero.

The variance method is defined in `/dask/bag/core.py:1139-1143` and uses the reduction framework with `var_chunk` and `var_aggregate` functions.

Documentation: https://docs.dask.org/en/stable/bag-api.html#dask.bag.Bag.var

## Proposed Fix

```diff
--- a/dask/bag/chunk.py
+++ b/dask/bag/chunk.py
@@ -31,6 +31,13 @@ def var_chunk(seq):

 def var_aggregate(x, ddof):
     squares, totals, counts = list(zip(*x))
     x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
+
+    if n <= ddof:
+        # Match NumPy behavior: return inf for ddof >= n
+        return float('inf')
+
     result = (x2 / n) - (x / n) ** 2
+    # Ensure non-negative due to floating-point precision
+    result = max(0.0, result)
     return result * n / (n - ddof)
```