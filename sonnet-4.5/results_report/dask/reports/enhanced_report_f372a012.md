# Bug Report: dask.bag.chunk Variance Functions Produce Negative and Incorrect Results

**Target**: `dask.bag.chunk.var_chunk` and `dask.bag.chunk.var_aggregate`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The variance calculation functions in dask.bag use a numerically unstable algorithm that produces mathematically impossible negative variance values and incorrect results for datasets with large values or when all values are identical.

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

    assert aggregate_result >= 0, f"Variance cannot be negative, got {aggregate_result}"
    assert math.isclose(aggregate_result, expected_variance,
                       rel_tol=1e-9, abs_tol=1e-9), f"Variance mismatch: got {aggregate_result}, expected {expected_variance}"

if __name__ == "__main__":
    # Run the test
    test_var_chunk_aggregate()
```

<details>

<summary>
**Failing input**: `[61800.0, 61838.0, 61841.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 21, in <module>
    test_var_chunk_aggregate()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 6, in test_var_chunk_aggregate
    min_value=-1e6, max_value=1e6),

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 16, in test_var_chunk_aggregate
    assert math.isclose(aggregate_result, expected_variance,
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       rel_tol=1e-9, abs_tol=1e-9), f"Variance mismatch: got {aggregate_result}, expected {expected_variance}"
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Variance mismatch: got 348.2222218513489, expected 348.2222222222222
Falsifying example: test_var_chunk_aggregate(
    values=[61800.0, 61838.0, 61841.0],
)
```
</details>

## Reproducing the Bug

```python
from dask.bag.chunk import var_chunk, var_aggregate

# Test case 1: Identical negative values (should have variance = 0)
values1 = [-356335.16553451226, -356335.16553451226, -356335.16553451226]
chunk_result1 = var_chunk(values1)
variance1 = var_aggregate([chunk_result1], ddof=0)

print("Test case 1: Three identical negative values")
print(f"Input: {values1}")
print(f"Computed variance: {variance1}")
print(f"Expected variance: 0.0")
print(f"Result: {'FAIL - Negative variance!' if variance1 < 0 else 'FAIL - Non-zero variance for identical values' if variance1 != 0 else 'PASS'}")
print()

# Test case 2: Identical positive values (should have variance = 0)
values2 = [259284.59765625, 259284.59765625, 259284.59765625]
chunk_result2 = var_chunk(values2)
variance2 = var_aggregate([chunk_result2], ddof=0)

print("Test case 2: Three identical positive values")
print(f"Input: {values2}")
print(f"Computed variance: {variance2}")
print(f"Expected variance: 0.0")
print(f"Result: {'FAIL - Negative variance!' if variance2 < 0 else 'FAIL - Non-zero variance for identical values' if variance2 != 0 else 'PASS'}")
print()

# Test case 3: Very large identical values
values3 = [1e15] * 10
chunk_result3 = var_chunk(values3)
variance3 = var_aggregate([chunk_result3], ddof=0)

print("Test case 3: Ten identical very large values")
print(f"Input: [1e15] * 10")
print(f"Computed variance: {variance3}")
print(f"Expected variance: 0.0")
print(f"Result: {'FAIL - Negative variance!' if variance3 < 0 else 'FAIL - Non-zero variance for identical values' if variance3 != 0 else 'PASS'}")
print()

# Test case 4: Simple values with known variance
values4 = [1, 2, 3, 4, 5]
chunk_result4 = var_chunk(values4)
variance4 = var_aggregate([chunk_result4], ddof=0)
expected4 = 2.0  # Variance of [1,2,3,4,5] is 2

print("Test case 4: Simple sequence [1,2,3,4,5]")
print(f"Input: {values4}")
print(f"Computed variance: {variance4}")
print(f"Expected variance: {expected4}")
print(f"Result: {'PASS' if abs(variance4 - expected4) < 1e-10 else 'FAIL - Incorrect variance'}")
```

<details>

<summary>
Output demonstrating negative variance and incorrect results
</summary>
```
Test case 1: Three identical negative values
Input: [-356335.16553451226, -356335.16553451226, -356335.16553451226]
Computed variance: -3.0517578125e-05
Expected variance: 0.0
Result: FAIL - Negative variance!

Test case 2: Three identical positive values
Input: [259284.59765625, 259284.59765625, 259284.59765625]
Computed variance: 7.62939453125e-06
Expected variance: 0.0
Result: FAIL - Non-zero variance for identical values

Test case 3: Ten identical very large values
Input: [1e15] * 10
Computed variance: -140737488355328.0
Expected variance: 0.0
Result: FAIL - Negative variance!

Test case 4: Simple sequence [1,2,3,4,5]
Input: [1, 2, 3, 4, 5]
Computed variance: 2.0
Expected variance: 2.0
Result: PASS
```
</details>

## Why This Is A Bug

This violates fundamental mathematical properties of variance:

1. **Variance cannot be negative** - Variance is defined as the average of squared deviations from the mean. Since squares are always non-negative, their average cannot be negative. The implementation produces negative values like `-3.0517578125e-05` and `-140737488355328.0`.

2. **Identical values must have zero variance** - When all values are identical, every deviation from the mean is zero, so variance must be exactly zero. The implementation produces non-zero results for identical values.

3. **Numerical instability increases with value magnitude** - The errors become catastrophic for large values (1e15 produces variance of -140737488355328 instead of 0), making the function unusable for real-world data with large scales.

The bug stems from using the mathematically correct but numerically unstable formula: `Var(X) = E[X²] - E[X]²`. This formula suffers from catastrophic cancellation when the two terms are nearly equal (which happens with large values or small variance). When floating-point rounding errors occur, the subtraction can produce negative results.

## Relevant Context

The problematic code is in `/lib/python3.13/site-packages/dask/bag/chunk.py`:

```python
def var_chunk(seq):
    squares, total, n = 0.0, 0.0, 0
    for x in seq:
        squares += x**2
        total += x
        n += 1
    return squares, total, n

def var_aggregate(x, ddof):
    squares, totals, counts = list(zip(*x))
    x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
    result = (x2 / n) - (x / n) ** 2
    return result * n / (n - ddof)
```

The `var()` method in `dask/bag/core.py:1141` uses these functions via the `reduction()` method for distributed variance computation.

This is a well-known numerical analysis problem with an established solution: Welford's online algorithm, which avoids the catastrophic cancellation by computing variance incrementally.

## Proposed Fix

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