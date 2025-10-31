# Bug Report: dask.dataframe.Series.quantile Produces Mathematically Incorrect and Unordered Results

**Target**: `dask.dataframe.Series.quantile`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quantile()` method in dask.dataframe produces incorrect quantile values that violate the fundamental mathematical property that quantiles must be ordered (Q_p1 ≤ Q_p2 when p1 < p2). This results in incorrect statistical calculations with errors up to 100% compared to correct values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import dask.dataframe as dd

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-100, max_value=100), min_size=2, max_size=50))
@settings(max_examples=100)
def test_quantile_ordered(values):
    pdf = pd.DataFrame({'x': values})
    ddf = dd.from_pandas(pdf, npartitions=2)

    q25 = ddf['x'].quantile(0.25).compute()
    q50 = ddf['x'].quantile(0.50).compute()
    q75 = ddf['x'].quantile(0.75).compute()

    assert q25 <= q50 <= q75, \
        f"Quantiles not ordered: Q25={q25}, Q50={q50}, Q75={q75}"

if __name__ == "__main__":
    # Run the test
    test_quantile_ordered()
```

<details>

<summary>
**Failing input**: `[0.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 21, in <module>
    test_quantile_ordered()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 6, in test_quantile_ordered
    min_value=-100, max_value=100), min_size=2, max_size=50))

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 16, in test_quantile_ordered
    assert q25 <= q50 <= q75, \
           ^^^^^^^^^^^^^^^^^
AssertionError: Quantiles not ordered: Q25=-0.5, Q50=-1.0, Q75=0.0
Falsifying example: test_quantile_ordered(
    values=[0.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/27/hypo.py:17
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

# Test case from the bug report
values = [0.0, 0.0, 2.0, 3.0, 1.0]
pdf = pd.DataFrame({'x': values})
ddf = dd.from_pandas(pdf, npartitions=2)

print("Input data:", values)
print()

# Pandas quantiles (correct)
pandas_q25 = pdf['x'].quantile(0.25)
pandas_q50 = pdf['x'].quantile(0.50)
pandas_q75 = pdf['x'].quantile(0.75)

print("Pandas quantiles (correct):")
print(f"  Q25: {pandas_q25}")
print(f"  Q50: {pandas_q50}")
print(f"  Q75: {pandas_q75}")
print(f"  Ordering check (Q25 <= Q50 <= Q75): {pandas_q25 <= pandas_q50 <= pandas_q75}")
print()

# Dask quantiles (potentially incorrect)
dask_q25 = ddf['x'].quantile(0.25).compute()
dask_q50 = ddf['x'].quantile(0.50).compute()
dask_q75 = ddf['x'].quantile(0.75).compute()

print("Dask quantiles:")
print(f"  Q25: {dask_q25}")
print(f"  Q50: {dask_q50}")
print(f"  Q75: {dask_q75}")
print(f"  Ordering check (Q25 <= Q50 <= Q75): {dask_q25 <= dask_q50 <= dask_q75}")
print()

if dask_q25 > dask_q50:
    print(f"ERROR: Ordering violation! Q25 ({dask_q25}) > Q50 ({dask_q50})")
else:
    print("No ordering violation detected")

# Additional test cases mentioned in the report
print("\n" + "="*60)
print("Additional test cases:")
print("="*60)

test_cases = [
    [0, 1],
    [0, 100],
    [1, 2, 3, 4, 5]
]

for test_values in test_cases:
    print(f"\nTest case: {test_values}")

    pdf_test = pd.DataFrame({'x': test_values})
    ddf_test = dd.from_pandas(pdf_test, npartitions=2)

    pandas_q25_test = pdf_test['x'].quantile(0.25)
    pandas_q50_test = pdf_test['x'].quantile(0.50)

    dask_q25_test = ddf_test['x'].quantile(0.25).compute()
    dask_q50_test = ddf_test['x'].quantile(0.50).compute()

    print(f"  Pandas: Q25={pandas_q25_test:.2f}, Q50={pandas_q50_test:.2f}")
    print(f"  Dask:   Q25={dask_q25_test:.2f}, Q50={dask_q50_test:.2f}")

    if abs(pandas_q25_test - dask_q25_test) > 0.01 or abs(pandas_q50_test - dask_q50_test) > 0.01:
        print(f"  ⚠️ Discrepancy detected!")
```

<details>

<summary>
Output demonstrating ordering violation and incorrect quantile values
</summary>
```
Input data: [0.0, 0.0, 2.0, 3.0, 1.0]

Pandas quantiles (correct):
  Q25: 0.0
  Q50: 1.0
  Q75: 2.0
  Ordering check (Q25 <= Q50 <= Q75): True

Dask quantiles:
  Q25: 1.5
  Q50: 1.0
  Q75: 2.0
  Ordering check (Q25 <= Q50 <= Q75): False

ERROR: Ordering violation! Q25 (1.5) > Q50 (1.0)

============================================================
Additional test cases:
============================================================

Test case: [0, 1]
  Pandas: Q25=0.25, Q50=0.50
  Dask:   Q25=0.00, Q50=0.00
  ⚠️ Discrepancy detected!

Test case: [0, 100]
  Pandas: Q25=25.00, Q50=50.00
  Dask:   Q25=0.00, Q50=0.00
  ⚠️ Discrepancy detected!

Test case: [1, 2, 3, 4, 5]
  Pandas: Q25=2.00, Q50=3.00
  Dask:   Q25=1.50, Q50=2.00
  ⚠️ Discrepancy detected!
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Violates fundamental mathematical property**: Quantiles must satisfy Q_p1 ≤ Q_p2 for p1 < p2. The bug produces cases where Q25 > Q50, which is mathematically impossible by definition of quantiles.

2. **Produces incorrect absolute values**: Beyond ordering violations, the quantile values themselves are wrong. For example:
   - For input `[0.0, 0.0, 2.0, 3.0, 1.0]`: Dask returns Q25=1.5 when the correct value is 0.0 (a 100% error)
   - For input `[0, 100]`: Dask returns Q50=0.0 when the correct value is 50.0 (a 100% error)

3. **Silent data corruption**: The function returns incorrect results without any warning or error, meaning users will unknowingly use wrong statistical measures in their analysis.

4. **Contradicts documentation**: The dask documentation states that quantile calculations should produce "Approximate row-wise and precise column-wise quantiles" - but the results are neither approximate nor precise, they're mathematically incorrect.

5. **Inconsistent with pandas**: Dask is designed to be a parallel computing library that mirrors pandas API. Users expect `ddf.quantile()` to produce results consistent with `pdf.quantile()`, but it produces wildly different results.

## Relevant Context

The bug appears to be in the distributed quantile merging algorithm implemented in `/home/npc/miniconda/lib/python3.13/site-packages/dask/array/percentile.py:205` in the `merge_percentiles` function. This function attempts to combine quantile calculations from different data partitions.

The issue specifically manifests when:
- Data is split across multiple partitions (npartitions > 1)
- The data distribution is non-uniform across partitions
- The "dask" method is used (which is the default)

Key code locations:
- Main quantile implementation: `dask/dataframe/dask_expr/_collection.py:4499`
- Quantile expression class: `dask/dataframe/dask_expr/_quantile.py:16` (SeriesQuantile)
- Dask method implementation: `dask/dataframe/dask_expr/_quantile.py:105` (SeriesQuantileDask)
- Percentile merging logic: `dask/array/percentile.py:205` (merge_percentiles)

## Proposed Fix

The issue lies in how quantiles from different partitions are merged. The current implementation in `merge_percentiles` doesn't properly handle the case where partitions have different data distributions. A high-level fix would involve:

1. **Ensure proper sorting and weighting**: The merge algorithm should properly weight quantile values based on the number of elements in each partition and ensure the final result maintains quantile ordering properties.

2. **Add validation**: Add a check after merging to ensure Q_p1 ≤ Q_p2 for all p1 < p2, and either correct violations or raise an error.

3. **Consider alternative algorithms**: The current custom algorithm has fundamental issues. Consider using established distributed quantile algorithms like:
   - T-Digest (already partially supported via the 'tdigest' method)
   - Q-Digest
   - GK-summary (Greenwald-Khanna algorithm)

As a temporary workaround for users:

```python
# Option 1: Use tdigest method if available
result = ddf['column'].quantile(q, method='tdigest')

# Option 2: Compute on full dataset (loses distributed benefits)
result = ddf['column'].compute().quantile(q)

# Option 3: Use single partition (if data fits in memory)
ddf_single = ddf.repartition(npartitions=1)
result = ddf_single['column'].quantile(q)
```