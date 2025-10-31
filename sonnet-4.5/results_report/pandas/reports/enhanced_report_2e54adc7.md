# Bug Report: pandas.core.indexers.utils.length_of_indexer Returns Negative Lengths for Negative Step Slices

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function incorrectly returns negative values when calculating the length of slices with negative steps, violating the mathematical definition of "length" and the function's documented contract to return "the expected length of target[indexer]".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from pandas.core.indexers.utils import length_of_indexer


@given(
    st.one_of(
        st.integers(min_value=0, max_value=100),
        st.builds(
            slice,
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            st.one_of(st.none(), st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0)),
        ),
        st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
        st.lists(st.booleans(), min_size=1, max_size=50),
    ),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
@example(indexer=slice(None, None, -1), target_len=1)  # Force testing the reported failing case
def test_length_of_indexer_matches_actual_length(indexer, target_len):
    target = list(range(target_len))

    if isinstance(indexer, list) and len(indexer) > 0 and isinstance(indexer[0], bool):
        if len(indexer) != target_len:
            return

    try:
        claimed_length = length_of_indexer(indexer, target)
        actual_result = target[indexer]
        actual_length = len(actual_result)
        assert claimed_length == actual_length, f"Claimed {claimed_length}, actual {actual_length} for indexer={indexer}, target_len={target_len}"
    except (IndexError, ValueError, TypeError, KeyError):
        pass


if __name__ == "__main__":
    test_length_of_indexer_matches_actual_length()
```

<details>

<summary>
**Failing input**: `indexer=slice(None, None, -1), target_len=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 38, in <module>
    test_length_of_indexer_matches_actual_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 6, in test_length_of_indexer_matches_actual_length
    st.one_of(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 32, in test_length_of_indexer_matches_actual_length
    assert claimed_length == actual_length, f"Claimed {claimed_length}, actual {actual_length} for indexer={indexer}, target_len={target_len}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Claimed -1, actual 1 for indexer=slice(None, None, -1), target_len=1
Falsifying explicit example: test_length_of_indexer_matches_actual_length(
    indexer=slice(None, None, -1),
    target_len=1,
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

# Test case 1: Basic negative step slice that should reverse a list
target = [0]  # Single element list
indexer = slice(None, None, -1)  # Standard Python idiom for reversing

claimed_length = length_of_indexer(indexer, target)
actual_result = target[indexer]  # This would be [0] in normal Python
actual_length = len(actual_result)

print("Test case 1: Single element list with reverse slice")
print(f"Target: {target}")
print(f"Indexer: {indexer}")
print(f"Actual result of target[indexer]: {actual_result}")
print(f"Actual length: {actual_length}")
print(f"Claimed length from length_of_indexer: {claimed_length}")
print(f"Match: {actual_length == claimed_length}")
print()

# Test case 2: Multi-element list with reverse slice
target2 = [0, 1, 2, 3, 4]
indexer2 = slice(None, None, -1)

claimed_length2 = length_of_indexer(indexer2, target2)
actual_result2 = target2[indexer2]
actual_length2 = len(actual_result2)

print("Test case 2: Multi-element list with reverse slice")
print(f"Target: {target2}")
print(f"Indexer: {indexer2}")
print(f"Actual result of target[indexer]: {actual_result2}")
print(f"Actual length: {actual_length2}")
print(f"Claimed length from length_of_indexer: {claimed_length2}")
print(f"Match: {actual_length2 == claimed_length2}")
print()

# Test case 3: Step of -2
target3 = [0, 1, 2, 3, 4]
indexer3 = slice(None, None, -2)

claimed_length3 = length_of_indexer(indexer3, target3)
actual_result3 = target3[indexer3]
actual_length3 = len(actual_result3)

print("Test case 3: Multi-element list with step=-2")
print(f"Target: {target3}")
print(f"Indexer: {indexer3}")
print(f"Actual result of target[indexer]: {actual_result3}")
print(f"Actual length: {actual_length3}")
print(f"Claimed length from length_of_indexer: {claimed_length3}")
print(f"Match: {actual_length3 == claimed_length3}")
```

<details>

<summary>
Function returns negative lengths instead of actual lengths
</summary>
```
Test case 1: Single element list with reverse slice
Target: [0]
Indexer: slice(None, None, -1)
Actual result of target[indexer]: [0]
Actual length: 1
Claimed length from length_of_indexer: -1
Match: False

Test case 2: Multi-element list with reverse slice
Target: [0, 1, 2, 3, 4]
Indexer: slice(None, None, -1)
Actual result of target[indexer]: [4, 3, 2, 1, 0]
Actual length: 5
Claimed length from length_of_indexer: -5
Match: False

Test case 3: Multi-element list with step=-2
Target: [0, 1, 2, 3, 4]
Indexer: slice(None, None, -2)
Actual result of target[indexer]: [4, 2, 0]
Actual length: 3
Claimed length from length_of_indexer: -2
Match: False
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Mathematical Impossibility**: Lengths are by definition non-negative integers. A length cannot be -1, -5, or any negative value. The function's docstring explicitly states it returns "the expected length of target[indexer]", which must be ≥ 0.

2. **Common Python Pattern Broken**: The slice `slice(None, None, -1)` (equivalent to `[::-1]`) is the standard Python idiom for reversing sequences. This is not an edge case but a fundamental operation used throughout Python code.

3. **Internal Validation Failure**: The function is used by `check_setitem_lengths` (line 175 in the same file) to validate assignment operations. When this returns negative values, the validation logic fails incorrectly, potentially allowing invalid operations or rejecting valid ones.

4. **Clear Logic Error**: The bug stems from lines 303-315 where the function handles None values incorrectly for negative steps:
   - Lines 303-310: Applies defaults assuming positive step (None start→0, None stop→target_len)
   - Lines 313-315: Attempts to "fix" negative steps by swapping start/stop and negating step
   - This approach fundamentally misunderstands how Python handles None with negative steps

## Relevant Context

The `length_of_indexer` function is part of pandas' internal indexing machinery located in `pandas.core.indexers.utils`. While it's in a private module (not public API), it's critical for pandas' internal operations.

Python's slicing semantics for negative steps:
- For positive step: `start=None` means 0, `stop=None` means len(sequence)
- For negative step: `start=None` means len(sequence)-1, `stop=None` means "before the beginning"

The current implementation applies positive-step defaults then tries to transform them, which produces incorrect results. This affects any pandas operation that internally validates indexer lengths.

Documentation: https://pandas.pydata.org/docs/reference/internals.html#indexing-utilities
Source code: https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexers/utils.py#L290-L330

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -298,18 +298,30 @@ def length_of_indexer(indexer, target=None) -> int:
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
-        if start is None:
-            start = 0
-        elif start < 0:
-            start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
-        elif stop < 0:
-            stop += target_len
+
         if step is None:
             step = 1
-        elif step < 0:
+
+        if step < 0:
+            # For negative step, handle None values correctly BEFORE conversion
+            if start is None:
+                start = target_len - 1
+            elif start < 0:
+                start += target_len
+            if stop is None:
+                stop = -target_len - 1  # Before the beginning
+            elif stop < 0:
+                stop += target_len
+            # Now convert to positive step calculation
             start, stop = stop + 1, start + 1
             step = -step
+        else:
+            # For positive step
+            if start is None:
+                start = 0
+            elif start < 0:
+                start += target_len
+            if stop is None or stop > target_len:
+                stop = target_len
+            elif stop < 0:
+                stop += target_len
+
         return (stop - start + step - 1) // step
```