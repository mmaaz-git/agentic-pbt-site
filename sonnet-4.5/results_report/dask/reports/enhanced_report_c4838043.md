# Bug Report: dask.bag.Bag.var Division by Zero Error

**Target**: `dask.bag.Bag.var`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Bag.var()` method crashes with `ZeroDivisionError` when the degrees of freedom parameter `ddof` equals or exceeds the number of elements in the bag, and also returns mathematically invalid negative variance when `ddof > n`.

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

if __name__ == "__main__":
    test_variance_no_crash()
```

<details>

<summary>
**Failing input**: `data=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 12, in test_variance_no_crash
    var = b.var(ddof=ddof).compute()
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

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 17, in <module>
    test_variance_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 8, in test_variance_no_crash
    def test_variance_no_crash(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 14, in test_variance_no_crash
    assert False, f"var() should not crash with ZeroDivisionError for ddof={ddof}, n={len(data)}"
           ^^^^^
AssertionError: var() should not crash with ZeroDivisionError for ddof=1, n=0
Falsifying example: test_variance_no_crash(
    data=[],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

# Test case 1: ddof equals n (both are 2)
print("Test case 1: [1.0, 2.0] with ddof=2")
b = db.from_sequence([1.0, 2.0], npartitions=1)
try:
    result = b.var(ddof=2).compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: Empty sequence
print("Test case 2: Empty sequence [] with ddof=0")
b = db.from_sequence([], npartitions=1)
try:
    result = b.var(ddof=0).compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: ddof > n
print("Test case 3: [1.0, 2.0, 3.0] with ddof=4")
b = db.from_sequence([1.0, 2.0, 3.0], npartitions=1)
try:
    result = b.var(ddof=4).compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ZeroDivisionError crashes and mathematically invalid negative variance
</summary>
```
Test case 1: [1.0, 2.0] with ddof=2
Error: ZeroDivisionError: float division by zero

==================================================

Test case 2: Empty sequence [] with ddof=0
Error: ZeroDivisionError: float division by zero

==================================================

Test case 3: [1.0, 2.0, 3.0] with ddof=4
Result: -2.000000000000001
```
</details>

## Why This Is A Bug

This violates expected behavior in three critical ways:

1. **Unhandled Division by Zero**: The variance calculation in `/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/chunk.py:36` divides by `(n - ddof)` without checking if this denominator is zero. When `ddof == n`, this causes an unhandled `ZeroDivisionError` crash rather than returning a mathematically appropriate value or raising an informative error.

2. **Empty Sequence Handling**: Line 35 in the same file (`result = (x2 / n) - (x / n) ** 2`) performs division by `n` which crashes when processing empty sequences (n=0), even with ddof=0.

3. **Mathematically Invalid Results**: When `ddof > n`, the function returns negative variance values (e.g., -2.0), which is mathematically impossible since variance by definition cannot be negative. This occurs because the denominator `(n - ddof)` becomes negative.

The Dask documentation for `var()` provides no guidance on handling these edge cases, while comparable libraries like NumPy and Pandas handle them gracefully - NumPy returns `inf` with a RuntimeWarning for division by zero and `nan` for empty arrays, while Pandas consistently returns `nan` for all invalid cases.

## Relevant Context

The issue stems from the variance aggregation function at `/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/chunk.py:32-36`:

```python
def var_aggregate(x, ddof):
    squares, totals, counts = list(zip(*x))
    x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
    result = (x2 / n) - (x / n) ** 2  # Line 35: crashes when n=0
    return result * n / (n - ddof)    # Line 36: crashes when n=ddof, negative when ddof>n
```

The variance formula requires dividing by `(n - ddof)` where:
- `n` is the number of elements
- `ddof` is the "Delta Degrees of Freedom"

Statistically, having `ddof >= n` means having more constraints than data points, which is meaningless. However, rather than crashing or returning invalid results, the function should handle these cases gracefully.

Related documentation:
- NumPy's variance: https://numpy.org/doc/stable/reference/generated/numpy.var.html
- Pandas' variance: https://pandas.pydata.org/docs/reference/api/pandas.Series.var.html

## Proposed Fix

```diff
--- a/dask/bag/chunk.py
+++ b/dask/bag/chunk.py
@@ -32,6 +32,10 @@ def var_chunk(seq):
 def var_aggregate(x, ddof):
     squares, totals, counts = list(zip(*x))
     x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
+    if n == 0:
+        raise ValueError("var() cannot be computed on an empty sequence")
+    if n <= ddof:
+        raise ValueError(f"degrees of freedom (ddof={ddof}) must be less than the number of elements (n={n})")
     result = (x2 / n) - (x / n) ** 2
     return result * n / (n - ddof)
```