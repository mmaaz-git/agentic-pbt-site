# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Invalid Bounds with Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds` returns invalid window bounds where `end[i] < start[i]` when initialized with a negative `window_size`, violating the fundamental invariant that window boundaries should always satisfy `start[i] <= end[i]`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.indexers import FixedForwardWindowIndexer


@settings(max_examples=1000)
@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_fixed_forward_window_start_le_end_invariant(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window bounds at index {i}: start={start[i]}, end={end[i]}"


if __name__ == "__main__":
    # Run the test
    test_fixed_forward_window_start_le_end_invariant()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1, step=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 22, in <module>
    test_fixed_forward_window_start_le_end_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 7, in test_fixed_forward_window_start_le_end_invariant
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 17, in test_fixed_forward_window_start_le_end_invariant
    assert start[i] <= end[i], f"Invalid window bounds at index {i}: start={start[i]}, end={end[i]}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Invalid window bounds at index 1: start=1, end=0
Falsifying example: test_fixed_forward_window_start_le_end_invariant(
    num_values=2,
    window_size=-1,
    step=1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Test with negative window size
indexer = FixedForwardWindowIndexer(window_size=-2)
start, end = indexer.get_window_bounds(num_values=5)

print(f"window_size=-2, num_values=5")
print(f"start: {start}")
print(f"end: {end}")
print()

# Check for invalid bounds
print("Invalid window bounds (where start > end):")
invalid_found = False
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"  Index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
        invalid_found = True

if not invalid_found:
    print("  None found")
print()

# Also test the exact failing case from hypothesis
print("Testing hypothesis failing case: window_size=-1, num_values=2")
indexer2 = FixedForwardWindowIndexer(window_size=-1)
start2, end2 = indexer2.get_window_bounds(num_values=2, step=1)
print(f"start: {start2}")
print(f"end: {end2}")

print("\nChecking bounds:")
for i in range(len(start2)):
    valid = "✓" if start2[i] <= end2[i] else "✗ INVALID"
    print(f"  Index {i}: start={start2[i]}, end={end2[i]} {valid}")
```

<details>

<summary>
Invalid window bounds detected with negative window_size
</summary>
```
window_size=-2, num_values=5
start: [0 1 2 3 4]
end: [0 0 0 1 2]

Invalid window bounds (where start > end):
  Index 1: start[1]=1 > end[1]=0
  Index 2: start[2]=2 > end[2]=0
  Index 3: start[3]=3 > end[3]=1
  Index 4: start[4]=4 > end[4]=2

Testing hypothesis failing case: window_size=-1, num_values=2
start: [0 1]
end: [0 0]

Checking bounds:
  Index 0: start=0, end=0 ✓
  Index 1: start=1, end=0 ✗ INVALID
```
</details>

## Why This Is A Bug

Window boundaries must always satisfy the invariant `start[i] <= end[i]` for all indices. This is a fundamental requirement for any window or range concept, as it defines a valid interval. When `FixedForwardWindowIndexer` is initialized with a negative `window_size`, the implementation in `get_window_bounds` (lines 340-341 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py`) computes:

```python
start = np.arange(0, num_values, step, dtype="int64")
end = start + self.window_size  # When window_size < 0, end < start!
```

This produces `end < start` for negative window sizes. The subsequent clipping operation on line 343 (`np.clip(end, 0, num_values)`) doesn't fix the fundamental issue of invalid bounds—it only ensures the values stay within the array range.

The class name "FixedForwardWindowIndexer" explicitly indicates forward-looking behavior, which is semantically incompatible with negative window sizes. All examples in the documentation use positive window sizes, and the concept of a "forward window" inherently implies looking ahead (positive direction).

When these invalid bounds are used with pandas rolling operations, they would produce unexpected NaN values or incorrect results rather than raising a clear error, potentially leading to silent data corruption.

## Relevant Context

The bug is located in the `FixedForwardWindowIndexer.get_window_bounds` method at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py:323-345`.

The class docstring and examples (lines 298-320) show only positive window_size values, suggesting negative values were not intended to be supported. However, there's no explicit validation to prevent negative values.

The parent class `BaseIndexer.__init__` (lines 72-79) also doesn't validate the window_size parameter, accepting any integer value.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html

## Proposed Fix

Add validation to reject negative window sizes in `FixedForwardWindowIndexer.get_window_bounds`:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -328,6 +328,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError("window_size must be non-negative for forward-looking windows")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```