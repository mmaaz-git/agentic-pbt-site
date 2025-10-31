# Bug Report: dask.dataframe.dask_expr._groupby Returns Negative Partition Count for Empty Groupby

**Target**: `dask.dataframe.dask_expr._groupby._adjust_split_out_for_group_keys`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The internal function `_adjust_split_out_for_group_keys` returns negative or zero values when called with an empty `by` list, violating the invariant that partition counts must be positive integers.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math

def _adjust_split_out_for_group_keys(npartitions, by):
    if len(by) == 1:
        return math.ceil(npartitions / 15)
    return math.ceil(npartitions / (10 / (len(by) - 1)))

@given(
    st.integers(min_value=1, max_value=1000),
    st.lists(st.text(), max_size=10)
)
def test_split_out_is_positive(npartitions, by):
    result = _adjust_split_out_for_group_keys(npartitions, by)
    assert result > 0, f"Expected positive split_out, got {result}"

if __name__ == "__main__":
    # Run the test
    test_split_out_is_positive()
```

<details>

<summary>
**Failing input**: `npartitions=1, by=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 19, in <module>
    test_split_out_is_positive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 10, in test_split_out_is_positive
    st.integers(min_value=1, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 15, in test_split_out_is_positive
    assert result > 0, f"Expected positive split_out, got {result}"
           ^^^^^^^^^^
AssertionError: Expected positive split_out, got 0
Falsifying example: test_split_out_is_positive(
    npartitions=1,
    by=[],
)
```
</details>

## Reproducing the Bug

```python
import math

def _adjust_split_out_for_group_keys(npartitions, by):
    if len(by) == 1:
        return math.ceil(npartitions / 15)
    return math.ceil(npartitions / (10 / (len(by) - 1)))

# Test case that crashes
npartitions = 100
by = []

print("Testing _adjust_split_out_for_group_keys with empty by list:")
print(f"npartitions = {npartitions}")
print(f"by = {by}")
print(f"len(by) = {len(by)}")
print()

try:
    result = _adjust_split_out_for_group_keys(npartitions, by)
    print(f"Result: {result}")

    # Show the calculation step by step
    print("\nStep-by-step calculation:")
    print(f"len(by) - 1 = {len(by)} - 1 = {len(by) - 1}")
    denominator = 10 / (len(by) - 1)
    print(f"10 / (len(by) - 1) = 10 / {len(by) - 1} = {denominator}")
    print(f"npartitions / denominator = {npartitions} / {denominator} = {npartitions / denominator}")
    print(f"math.ceil({npartitions / denominator}) = {math.ceil(npartitions / denominator)}")

    # Check if the result makes sense
    print(f"\nIs result positive? {result > 0}")
    print(f"Is result a valid number of partitions? {result > 0 and isinstance(result, int)}")

except Exception as e:
    print(f"Error: {e}")
```

<details>

<summary>
Result: -10 (negative partition count)
</summary>
```
Testing _adjust_split_out_for_group_keys with empty by list:
npartitions = 100
by = []
len(by) = 0

Result: -10

Step-by-step calculation:
len(by) - 1 = 0 - 1 = -1
10 / (len(by) - 1) = 10 / -1 = -10.0
npartitions / denominator = 100 / -10.0 = -10.0
math.ceil(-10.0) = -10

Is result positive? False
Is result a valid number of partitions? False
```
</details>

## Why This Is A Bug

This violates the fundamental invariant that partition counts must be positive integers. When `len(by) == 0`, the formula `npartitions / (10 / (len(by) - 1))` becomes `npartitions / (10 / -1)`, resulting in negative values. The function is used internally by dask's groupby operations at two locations:

1. `GroupByApplyConcatApply._tune_down` (line 240)
2. `GroupByReduction._tune_down` (line 692)

Both call sites use `functools.partial(_adjust_split_out_for_group_keys, by=self.by)` to create a function for determining output partitions. While groupby operations with empty `by` lists are invalid in both pandas and dask (pandas raises `ValueError: No group keys passed!`), internal functions should still handle edge cases gracefully rather than producing mathematically nonsensical results.

## Relevant Context

The function is located in `/lib/python3.13/site-packages/dask/dataframe/dask_expr/_groupby.py` at line 95. It's an internal utility function (underscore prefix) used to calculate the number of output partitions for groupby split operations based on the number of grouping keys.

The function implements a heuristic:
- 1 grouping key: divides partitions by 15
- Multiple keys: uses formula `10 / (len(by) - 1)` as divisor

This heuristic likely optimizes performance based on expected data distribution patterns with different numbers of grouping columns. However, it fails to validate that `by` is non-empty, leading to division by negative numbers when `len(by) == 0`.

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/_groupby.py
+++ b/dask/dataframe/dask_expr/_groupby.py
@@ -94,6 +94,8 @@ def _as_dict(key, value):


 def _adjust_split_out_for_group_keys(npartitions, by):
+    if len(by) == 0:
+        raise ValueError("Cannot adjust split_out for empty 'by' list")
     if len(by) == 1:
         return math.ceil(npartitions / 15)
     return math.ceil(npartitions / (10 / (len(by) - 1)))
```