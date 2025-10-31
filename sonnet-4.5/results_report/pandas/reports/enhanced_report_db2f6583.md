# Bug Report: FixedForwardWindowIndexer Accepts Negative Window Size Leading to Invalid Bounds

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer` incorrectly accepts negative `window_size` values and produces mathematically invalid window bounds where `start[i] > end[i]`, causing rolling operations to silently return incorrect results (all zeros) instead of raising an error.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=-1),
)
@settings(max_examples=500)
def test_fixed_forward_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"start[{i}]={start[i]} > end[{i}]={end[i]}"

if __name__ == "__main__":
    test_fixed_forward_negative_window_size()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 18, in <module>
    test_fixed_forward_negative_window_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 6, in test_fixed_forward_negative_window_size
    num_values=st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 15, in test_fixed_forward_negative_window_size
    assert start[i] <= end[i], f"start[{i}]={start[i]} > end[{i}]={end[i]}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: start[1]=1 > end[1]=0
Falsifying example: test_fixed_forward_negative_window_size(
    num_values=2,
    window_size=-1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

# Create an indexer with negative window size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 3 values
start, end = indexer.get_window_bounds(num_values=3)

print("Window bounds for num_values=3, window_size=-1:")
print(f"start array: {start}")
print(f"end array: {end}")
print()

# Check the invariant violation
for i in range(len(start)):
    print(f"Window {i}: start[{i}]={start[i]}, end[{i}]={end[i]}")
    if start[i] > end[i]:
        print(f"  *** INVARIANT VIOLATED: start[{i}] > end[{i}] ***")
    else:
        print(f"  OK: start[{i}] <= end[{i}]")

# Demonstrate the effect on rolling operations
import numpy as np
df = pd.DataFrame({'A': [1, 2, 3]})
print("\nDataFrame:")
print(df)

# Custom rolling with the broken indexer
print("\nRolling with FixedForwardWindowIndexer(window_size=-1):")
rolling = df.rolling(indexer)
result = rolling.sum()
print(result)
```

<details>

<summary>
Output shows invalid bounds and incorrect rolling results
</summary>
```
Window bounds for num_values=3, window_size=-1:
start array: [0 1 2]
end array: [0 0 1]

Window 0: start[0]=0, end[0]=0
  OK: start[0] <= end[0]
Window 1: start[1]=1, end[1]=0
  *** INVARIANT VIOLATED: start[1] > end[1] ***
Window 2: start[2]=2, end[2]=1
  *** INVARIANT VIOLATED: start[2] > end[2] ***

DataFrame:
   A
0  1
1  2
2  3

Rolling with FixedForwardWindowIndexer(window_size=-1):
     A
0  0.0
1  0.0
2  0.0
```
</details>

## Why This Is A Bug

1. **Violates fundamental indexing invariant**: The condition `start[i] <= end[i]` must hold for all valid window bounds as this is required for array slicing semantics. When `start[i] > end[i]`, the slice `data[start[i]:end[i]]` produces an empty array.

2. **Produces silently incorrect results**: Rolling operations with negative window sizes return all zeros (or NaN for operations like mean) instead of raising a clear error. This silent data corruption is particularly dangerous in data analysis pipelines.

3. **Semantic contradiction**: The class name "FixedForwardWindowIndexer" explicitly indicates a "forward-looking" window. A negative window size contradicts this semantic meaning - you cannot look forward by a negative amount.

4. **Inconsistent with pandas conventions**: The standard `pd.DataFrame.rolling(window=n)` method properly validates that `window >= 0` and raises `ValueError: window must be an integer 0 or greater` for negative values. The low-level indexer should maintain the same constraint.

5. **Implementation clearly expects non-negative values**: Looking at the source code in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py` (lines 340-345), the implementation adds `window_size` to the start positions and clips to valid bounds, which only makes sense for non-negative window sizes.

## Relevant Context

The FixedForwardWindowIndexer is part of pandas' custom indexer API introduced to support more flexible rolling window operations. The class is located in `pandas/core/indexers/objects.py` and inherits from `BaseIndexer`.

Key observations from the source code:
- Line 341: `end = start + self.window_size` - when window_size is negative, this produces end values less than start
- Line 343: The clipping logic only applies when `self.window_size` is truthy (non-zero), but doesn't check for negative values
- The documentation and examples (lines 303-320) only show positive window_size values
- No validation is performed in `__init__` or `get_window_bounds` methods

Related pandas documentation:
- [Custom window rolling](https://pandas.pydata.org/docs/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html)
- [Rolling window operations](https://pandas.pydata.org/docs/user_guide/window.html)

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -328,6 +328,9 @@ class FixedForwardWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError(
+                f"window_size must be non-negative, got {self.window_size}")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```