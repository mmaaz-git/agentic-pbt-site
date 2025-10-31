# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length for Empty Ranges

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for empty `range` objects (where `stop <= start` with positive step), violating its documented contract and Python's standard range semantics which require length 0 for empty ranges.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_length_of_indexer_range(start, stop, step):
    r = range(start, stop, step)
    result = indexers.length_of_indexer(r)
    expected = len(r)
    assert result == expected, f"length_of_indexer(range({start}, {stop}, {step})) returned {result}, expected {expected}"

if __name__ == "__main__":
    test_length_of_indexer_range()
```

<details>

<summary>
**Failing input**: `start=1, stop=0, step=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 17, in <module>
    test_length_of_indexer_range()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 5, in test_length_of_indexer_range
    start=st.integers(min_value=0, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 14, in test_length_of_indexer_range
    assert result == expected, f"length_of_indexer(range({start}, {stop}, {step})) returned {result}, expected {expected}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: length_of_indexer(range(1, 0, 1)) returned -1, expected 0
Falsifying example: test_length_of_indexer_range(
    start=1,
    stop=0,
    step=1,
)
```
</details>

## Reproducing the Bug

```python
import pandas.core.indexers as indexers

# Test case 1: range(1, 0, 1)
r = range(1, 0, 1)
print(f"Test 1: range(1, 0, 1)")
print(f"  Python len(range(1, 0, 1)) = {len(r)}")
print(f"  pandas length_of_indexer(range(1, 0, 1)) = {indexers.length_of_indexer(r)}")
print()

# Test case 2: range(10, 0, 2)
r = range(10, 0, 2)
print(f"Test 2: range(10, 0, 2)")
print(f"  Python len(range(10, 0, 2)) = {len(r)}")
print(f"  pandas length_of_indexer(range(10, 0, 2)) = {indexers.length_of_indexer(r)}")
print()

# Test case 3: range(5, 5, 1) - equal start and stop
r = range(5, 5, 1)
print(f"Test 3: range(5, 5, 1) [equal start/stop]")
print(f"  Python len(range(5, 5, 1)) = {len(r)}")
print(f"  pandas length_of_indexer(range(5, 5, 1)) = {indexers.length_of_indexer(r)}")
print()

# Test case 4: range(100, 50, 3) - larger range
r = range(100, 50, 3)
print(f"Test 4: range(100, 50, 3)")
print(f"  Python len(range(100, 50, 3)) = {len(r)}")
print(f"  pandas length_of_indexer(range(100, 50, 3)) = {indexers.length_of_indexer(r)}")
print()

# Test case 5: range(0, 10, 1) - normal valid range for comparison
r = range(0, 10, 1)
print(f"Test 5: range(0, 10, 1) [normal range for comparison]")
print(f"  Python len(range(0, 10, 1)) = {len(r)}")
print(f"  pandas length_of_indexer(range(0, 10, 1)) = {indexers.length_of_indexer(r)}")
```

<details>

<summary>
Output showing negative values returned for empty ranges
</summary>
```
Test 1: range(1, 0, 1)
  Python len(range(1, 0, 1)) = 0
  pandas length_of_indexer(range(1, 0, 1)) = -1

Test 2: range(10, 0, 2)
  Python len(range(10, 0, 2)) = 0
  pandas length_of_indexer(range(10, 0, 2)) = -5

Test 3: range(5, 5, 1) [equal start/stop]
  Python len(range(5, 5, 1)) = 0
  pandas length_of_indexer(range(5, 5, 1)) = 0

Test 4: range(100, 50, 3)
  Python len(range(100, 50, 3)) = 0
  pandas length_of_indexer(range(100, 50, 3)) = -17

Test 5: range(0, 10, 1) [normal range for comparison]
  Python len(range(0, 10, 1)) = 10
  pandas length_of_indexer(range(0, 10, 1)) = 10
```
</details>

## Why This Is A Bug

The function `length_of_indexer` is documented as "Return the expected length of target[indexer]" (line 292 of `/pandas/core/indexers/utils.py`). For `range` objects, this means it should return the same value as Python's built-in `len()` function, as the expected length of indexing with a range is the number of elements that range contains.

When a range is empty (produces no values during iteration), Python's `len()` correctly returns 0. However, `length_of_indexer` returns negative values in these cases, violating several principles:

1. **Violates documented contract**: The function should return the "expected length" which can never be negative
2. **Violates Python semantics**: Python's `len(range(...))` never returns negative values
3. **Semantically incorrect**: Negative lengths have no meaning in the context of array/sequence indexing
4. **Inconsistent behavior**: The function correctly handles `range(5, 5, 1)` (returns 0) but fails for `range(6, 5, 1)` (returns -1)

The bug occurs at line 326 where the formula `(indexer.stop - indexer.start) // indexer.step` doesn't account for empty ranges where `stop < start` with positive step.

## Relevant Context

The `length_of_indexer` function is an internal utility in pandas located at `/pandas/core/indexers/utils.py`. While not part of the public API, it's used internally by pandas indexing operations (imported and used in `/pandas/core/indexing.py`).

The function handles multiple indexer types including slices, arrays, and ranges. The bug specifically affects the range handling branch. The current implementation uses a simple arithmetic formula that works correctly for non-empty ranges but produces negative results for empty ones.

Python's range objects have well-defined length semantics documented at https://docs.python.org/3/library/stdtypes.html#range. An empty range (one that yields no elements) always has length 0, regardless of the reason it's empty.

## Proposed Fix

The simplest and most correct fix is to delegate to Python's built-in `len()` function:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -323,7 +323,7 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        return len(indexer)
     elif not is_list_like_indexer(indexer):
         return 1
     raise AssertionError("cannot find the length of the indexer")
```