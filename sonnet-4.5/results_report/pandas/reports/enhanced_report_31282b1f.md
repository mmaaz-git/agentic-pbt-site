# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Accepts Invalid Negative Window Sizes

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` accepts negative `window_size` values and produces mathematically invalid window bounds where `start[i] > end[i]`, violating the fundamental invariant that window start indices must be less than or equal to end indices for valid array slicing.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=-10, max_value=-1))
def test_fixed_forward_window_negative_produces_invalid_bounds(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Window bounds violated: start[{i}]={start[i]} > end[{i}]={end[i]}"


if __name__ == "__main__":
    # Run the property test
    test_fixed_forward_window_negative_produces_invalid_bounds()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 17, in <module>
    test_fixed_forward_window_negative_produces_invalid_bounds()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 7, in test_fixed_forward_window_negative_produces_invalid_bounds
    def test_fixed_forward_window_negative_produces_invalid_bounds(num_values, window_size):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 12, in test_fixed_forward_window_negative_produces_invalid_bounds
    assert start[i] <= end[i], f"Window bounds violated: start[{i}]={start[i]} > end[{i}]={end[i]}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Window bounds violated: start[1]=1 > end[1]=0
Falsifying example: test_fixed_forward_window_negative_produces_invalid_bounds(
    num_values=2,
    window_size=-1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Create indexer with negative window_size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 2 values
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print()

# Check if bounds are valid
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Invalid bounds at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
    else:
        print(f"Valid bounds at index {i}: start[{i}]={start[i]} <= end[{i}]={end[i]}")
```

<details>

<summary>
Invalid window bounds produced with negative window_size
</summary>
```
start: [0 1]
end: [0 0]

Valid bounds at index 0: start[0]=0 <= end[0]=0
Invalid bounds at index 1: start[1]=1 > end[1]=0
```
</details>

## Why This Is A Bug

The class `FixedForwardWindowIndexer` is explicitly designed for forward-looking windows, as indicated by its name and documentation. The class documentation states it "Creates window boundaries for fixed-length windows that include the current row," and the example shows windows that look forward from each row (e.g., row 0 includes rows 0 and 1 with window_size=2).

When a negative `window_size` is provided, the implementation in `get_window_bounds()` produces mathematically invalid results:

1. **Line 340-341** in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py`:
   ```python
   start = np.arange(0, num_values, step, dtype="int64")
   end = start + self.window_size
   ```
   With `window_size=-1`, this produces `end = [0, 1] + (-1) = [-1, 0]`

2. **Line 342-343** applies clipping only if window_size is truthy (non-zero):
   ```python
   if self.window_size:
       end = np.clip(end, 0, num_values)
   ```
   This clips negative end values to 0, resulting in `end = [0, 0]`

3. The final bounds are `start = [0, 1]` and `end = [0, 0]`, creating an invalid window at index 1 where `start[1]=1 > end[1]=0`.

This violates the fundamental invariant of array indexing where `start <= end`. Any operation using these bounds would produce incorrect results or fail. For example:
- Array slicing `array[start[1]:end[1]]` becomes `array[1:0]`, which is an empty slice
- Rolling window operations would compute over incorrect data ranges
- Iteration logic expecting `start <= end` would fail or require special handling

## Relevant Context

The `FixedForwardWindowIndexer` class inherits from `BaseIndexer` and is part of the public API exported in `pandas.api.indexers`. Other indexer classes in the same module (e.g., `FixedWindowIndexer`, `VariableOffsetWindowIndexer`) have different validation logic, but none explicitly validate for negative window sizes.

The semantic meaning of "forward" in the class name strongly implies that negative window sizes are nonsensical - a forward-looking window cannot look backward. The lack of input validation appears to be an oversight rather than intentional support for negative values.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html
Source code: pandas/core/indexers/objects.py:297-345

## Proposed Fix

Add validation in the `get_window_bounds` method to reject negative window sizes:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -329,6 +329,9 @@ class FixedForwardWindowIndexer(BaseIndexer):
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
+        if self.window_size < 0:
+            raise ValueError(
+                f"Forward-looking windows require non-negative window_size, got {self.window_size}"
+            )
         if step is None:
             step = 1
```