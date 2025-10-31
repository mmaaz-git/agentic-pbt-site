# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Lengths for Negative Step Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for slices with negative steps when `start` or `stop` are `None`, violating the mathematical constraint that lengths must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=100),
)
def test_length_of_indexer_matches_actual_length(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = list(range(target_len))

    expected_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert expected_length == actual_length, f"For slice({start}, {stop}, {step}) on target of length {target_len}: expected {expected_length} but actual is {actual_length}"

if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_matches_actual_length()
```

<details>

<summary>
**Failing input**: `slice(1, None, None) on target of length 0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 21, in <module>
    test_length_of_indexer_matches_actual_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 5, in test_length_of_indexer_matches_actual_length
    start=st.integers(min_value=-100, max_value=100) | st.none(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 17, in test_length_of_indexer_matches_actual_length
    assert expected_length == actual_length, f"For slice({start}, {stop}, {step}) on target of length {target_len}: expected {expected_length} but actual is {actual_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: For slice(1, None, None) on target of length 0: expected -1 but actual is 0
Falsifying example: test_length_of_indexer_matches_actual_length(
    start=1,
    stop=None,
    step=None,
    target_len=0,
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

# Test case 1: Simple list with negative step
slc = slice(None, None, -1)
target = [0]

expected = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Test 1: slice(None, None, -1) on [0]")
print(f"length_of_indexer: {expected}")
print(f"actual: {actual}")
print(f"target[slc]: {target[slc]}")
print()

# Test case 2: Longer list with negative step
target2 = [0, 1, 2, 3, 4]
slc2 = slice(None, None, -1)

expected2 = length_of_indexer(slc2, target2)
actual2 = len(target2[slc2])

print(f"Test 2: slice(None, None, -1) on [0, 1, 2, 3, 4]")
print(f"length_of_indexer: {expected2}")
print(f"actual: {actual2}")
print(f"target[slc]: {target2[slc2]}")
print()

# Test case 3: With step -2
slc3 = slice(None, None, -2)

expected3 = length_of_indexer(slc3, target2)
actual3 = len(target2[slc3])

print(f"Test 3: slice(None, None, -2) on [0, 1, 2, 3, 4]")
print(f"length_of_indexer: {expected3}")
print(f"actual: {actual3}")
print(f"target[slc]: {target2[slc3]}")
```

<details>

<summary>
Returns negative lengths for slices with negative steps
</summary>
```
Test 1: slice(None, None, -1) on [0]
length_of_indexer: -1
actual: 1
target[slc]: [0]

Test 2: slice(None, None, -1) on [0, 1, 2, 3, 4]
length_of_indexer: -5
actual: 5
target[slc]: [4, 3, 2, 1, 0]

Test 3: slice(None, None, -2) on [0, 1, 2, 3, 4]
length_of_indexer: -2
actual: 3
target[slc]: [4, 2, 0]
```
</details>

## Why This Is A Bug

The `length_of_indexer` function is documented in its docstring to "Return the expected length of target[indexer]". A length, by definition, must be a non-negative integer representing a count of elements.

When given slices with negative steps (like `slice(None, None, -1)` which reverses a sequence), the function returns negative values. For example:
- `slice(None, None, -1)` on `[0]` returns `-1`, but `[0][::-1]` has length 1
- `slice(None, None, -1)` on `[0,1,2,3,4]` returns `-5`, but the actual reversed list has length 5

The root cause is in the handling of None values for start/stop with negative steps (lines 303-316 in `/pandas/core/indexers/utils.py`). The function incorrectly defaults `start=0` and `stop=target_len` when these are None, which are the correct defaults for positive steps but wrong for negative steps. For negative steps in Python slicing:
- `start=None` means start from the last element (index `target_len - 1`)
- `stop=None` means go all the way to (and including) the first element

This bug affects data validation in pandas, particularly in `check_setitem_lengths` (line 175 of the same file), which uses `length_of_indexer` to ensure array assignments have compatible lengths. With negative lengths being returned, this validation could incorrectly reject valid assignments or accept invalid ones.

## Relevant Context

The `length_of_indexer` function is a core utility in pandas indexing system located at `/pandas/core/indexers/utils.py`. It's used by `check_setitem_lengths` to validate array assignment operations.

Python's slice semantics for negative steps:
- `lst[::-1]` reverses the list
- `lst[::-2]` takes every second element in reverse
- When step is negative and start/stop are None, Python starts from the end and goes to the beginning

Documentation: The function's docstring at line 291-297 states it should "Return the expected length of target[indexer]" with return type `int`.

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -300,14 +300,23 @@ def length_of_indexer(indexer, target=None) -> int:
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
+
+        if step is None:
+            step = 1
+
+        # Handle None values differently for positive vs negative steps
         if start is None:
-            start = 0
+            start = target_len - 1 if step < 0 else 0
         elif start < 0:
             start += target_len
+
         if stop is None or stop > target_len:
-            stop = target_len
+            if step < 0:
+                stop = -1  # Will become 0 after the negative step adjustment
+            else:
+                stop = target_len
         elif stop < 0:
             stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
+
+        if step < 0:
             start, stop = stop + 1, start + 1
             step = -step
```