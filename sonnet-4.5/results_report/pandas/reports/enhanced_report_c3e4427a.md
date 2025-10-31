# Bug Report: pandas.api.indexers.length_of_indexer Returns Negative Length Values

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative values for certain slice configurations, violating the fundamental invariant that lengths must be non-negative and contradicting its documented behavior of returning "the expected length of target[indexer]".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.indexers.utils import length_of_indexer

@given(
    indexer=st.slices(20),
    target_len=st.integers(min_value=0, max_value=20),
)
def test_length_of_indexer_slice_matches_actual(indexer, target_len):
    target = list(range(target_len))

    expected_len = length_of_indexer(indexer, target)
    actual_result = target[indexer]
    actual_len = len(actual_result)

    assert expected_len == actual_len, (
        f"length_of_indexer({indexer}, len={target_len}) = {expected_len}, "
        f"but len(target[indexer]) = {actual_len}"
    )

if __name__ == "__main__":
    test_length_of_indexer_slice_matches_actual()
```

<details>

<summary>
**Failing input**: `indexer=slice(None, None, -1), target_len=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 21, in <module>
    test_length_of_indexer_slice_matches_actual()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 5, in test_length_of_indexer_slice_matches_actual
    indexer=st.slices(20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 15, in test_length_of_indexer_slice_matches_actual
    assert expected_len == actual_len, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: length_of_indexer(slice(None, None, -1), len=1) = -1, but len(target[indexer]) = 1
Falsifying example: test_length_of_indexer_slice_matches_actual(
    indexer=slice(None, None, -1),
    target_len=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/23/hypo.py:16

```
</details>

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

target = []
indexer = slice(0, -20, None)

result = length_of_indexer(indexer, target)
actual_length = len(target[indexer])

print(f"length_of_indexer returned: {result}")
print(f"Actual length: {actual_length}")
assert result == actual_length, f"Expected {actual_length}, but got {result}"
```

<details>

<summary>
AssertionError: Expected 0, but got -20
</summary>
```
length_of_indexer returned: -20
Actual length: 0
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/repo.py", line 11, in <module>
    assert result == actual_length, f"Expected {actual_length}, but got {result}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 0, but got -20

```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Violates Python's Length Invariant**: In Python, the `len()` function always returns non-negative integers. The docstring states the function should "Return the expected length of target[indexer]", which means it should return what `len(target[indexer])` would return.

2. **Mathematical Impossibility**: A length represents a count of elements, which by definition cannot be negative. Negative lengths have no meaningful interpretation in the context of sequence indexing.

3. **Incorrect Edge Case Handling**: When slicing an empty list with any indices (including negative ones like `slice(0, -20)`), Python always returns an empty list with length 0. The function fails to handle this correctly.

4. **Code Logic Error**: The bug occurs in lines 309-310 of the function. When `stop < 0`, the code adds the target length (`stop += target_len`). For an empty target (`target_len = 0`) with `stop = -20`, this results in `stop = -20 + 0 = -20`. The function then calculates `(stop - start + step - 1) // step = (-20 - 0 + 1 - 1) // 1 = -20`, never clamping the negative result.

## Relevant Context

The `length_of_indexer` function is located in `/pandas/core/indexers/utils.py` at line 290. It's an internal pandas utility function (not part of the public API) used for calculating expected lengths during indexing operations.

Key observations from the code:
- The function handles slices specially when a target is provided (lines 298-316)
- It attempts to normalize negative indices by adding the target length (lines 305-306, 309-310)
- It lacks bounds checking after normalizing negative indices, allowing negative values to propagate through

Python's slicing behavior documentation (https://docs.python.org/3/library/stdtypes.html) confirms that:
- Negative indices are interpreted as `len(s) + i`
- Slices always return valid subsequences (never negative length)
- Slicing an empty sequence with any indices returns an empty sequence

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -308,6 +308,8 @@ def length_of_indexer(indexer, target=None) -> int:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            if stop < 0:
+                stop = 0
         if step is None:
             step = 1
         elif step < 0:
```