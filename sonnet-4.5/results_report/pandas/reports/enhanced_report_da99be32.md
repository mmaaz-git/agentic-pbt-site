# Bug Report: pandas.core.indexers.utils.length_of_indexer Returns Negative Length for Empty Lists with Negative Slice Indices

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative values when computing the length of a slice with negative stop indices on empty or small sequences, violating the Python invariant that lengths must be non-negative.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@given(
    target_list=st.lists(st.integers(), min_size=0, max_size=50),
    slice_start=st.integers(min_value=-60, max_value=60) | st.none(),
    slice_stop=st.integers(min_value=-60, max_value=60) | st.none(),
    slice_step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
@settings(max_examples=500)
def test_length_of_indexer_slice(target_list, slice_start, slice_stop, slice_step):
    indexer = slice(slice_start, slice_stop, slice_step)
    expected_length = len(target_list[indexer])
    computed_length = length_of_indexer(indexer, target_list)
    assert computed_length == expected_length, \
        f"Mismatch for target_list={target_list}, indexer={indexer}: " \
        f"expected {expected_length}, got {computed_length}"

if __name__ == "__main__":
    test_length_of_indexer_slice()
```

<details>

<summary>
**Failing input**: `target_list=[], slice_start=None, slice_stop=-1, slice_step=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 23, in <module>
    test_length_of_indexer_slice()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 8, in test_length_of_indexer_slice
    target_list=st.lists(st.integers(), min_size=0, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 18, in test_length_of_indexer_slice
    assert computed_length == expected_length, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mismatch for target_list=[], indexer=slice(None, -1, None): expected 0, got -1
Falsifying example: test_length_of_indexer_slice(
    target_list=[],
    slice_start=None,
    slice_stop=-1,
    slice_step=None,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/5/hypo.py:19
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers import length_of_indexer

# Test case 1: Empty list with negative stop
target = []
indexer = slice(None, -1, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 1: Empty list with slice(None, -1, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ❌ MISMATCH: Expected {actual_length}, got {computed_length}")
print()

# Test case 2: Empty list with more negative stop
target = []
indexer = slice(None, -5, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 2: Empty list with slice(None, -5, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ❌ MISMATCH: Expected {actual_length}, got {computed_length}")
print()

# Test case 3: Small list with large negative stop
target = [1, 2]
indexer = slice(None, -5, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 3: List [1,2] with slice(None, -5, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ❌ MISMATCH: Expected {actual_length}, got {computed_length}")
print()

# Test case 4: Working case for comparison
target = [1, 2, 3]
indexer = slice(None, -1, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 4: List [1,2,3] with slice(None, -1, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ✓ CORRECT: Both are {actual_length}")
```

<details>

<summary>
Demonstrating negative length returns for empty and small lists
</summary>
```
Test 1: Empty list with slice(None, -1, None)
  target = []
  len(target[indexer]) = 0
  length_of_indexer(indexer, target) = -1
  ❌ MISMATCH: Expected 0, got -1

Test 2: Empty list with slice(None, -5, None)
  target = []
  len(target[indexer]) = 0
  length_of_indexer(indexer, target) = -5
  ❌ MISMATCH: Expected 0, got -5

Test 3: List [1,2] with slice(None, -5, None)
  target = [1, 2]
  len(target[indexer]) = 0
  length_of_indexer(indexer, target) = -3
  ❌ MISMATCH: Expected 0, got -3

Test 4: List [1,2,3] with slice(None, -1, None)
  target = [1, 2, 3]
  len(target[indexer]) = 2
  length_of_indexer(indexer, target) = 2
  ✓ CORRECT: Both are 2
```
</details>

## Why This Is A Bug

The function's docstring at line 292 explicitly states it returns "the expected length of target[indexer]", establishing a clear contract that the function should return the same value as `len(target[indexer])`.

This bug violates several fundamental principles:

1. **Python's length invariant**: In Python, the `len()` function always returns a non-negative integer. Sequence lengths cannot be negative - this is a core language guarantee. Returning -1 for a length operation violates this fundamental semantic.

2. **Contract violation**: The function promises to return what `len(target[indexer])` would return, but it returns different values. When Python evaluates `[][:-1]`, it correctly returns an empty list with length 0, not -1.

3. **Internal consistency**: This function is used internally in pandas validation logic (e.g., in `check_setitem_lengths` at line 175), where negative lengths could lead to incorrect validation behavior and potential downstream bugs.

4. **Mathematical correctness**: The length of any subset of a sequence must be between 0 and the original sequence length, inclusive. Negative lengths have no mathematical meaning in this context.

## Relevant Context

The bug occurs in the slice handling logic at lines 309-310 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py`:

```python
elif stop < 0:
    stop += target_len
```

When `stop` is negative and `|stop| > target_len`, the calculation `stop + target_len` produces a negative result. For example, with an empty list (`target_len=0`) and `stop=-1`, we get `stop = -1 + 0 = -1`. This negative value then propagates through to the final length calculation at line 316: `(stop - start + step - 1) // step`.

Python's built-in slicing handles this case correctly by clamping negative indices that go before the start of the sequence to 0. The pandas function should do the same.

The same issue exists for the `start` parameter at lines 305-306, though it's less commonly triggered.

This utility function is part of pandas' internal indexing infrastructure and is used in validation routines that check whether assignment operations have compatible shapes. Incorrect length calculations could lead to accepting invalid assignments or rejecting valid ones.

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -303,11 +303,11 @@ def length_of_indexer(indexer, target=None) -> int:
         if start is None:
             start = 0
         elif start < 0:
-            start += target_len
+            start = max(0, start + target_len)
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
-            stop += target_len
+            stop = max(0, stop + target_len)
         if step is None:
             step = 1
         elif step < 0:
```