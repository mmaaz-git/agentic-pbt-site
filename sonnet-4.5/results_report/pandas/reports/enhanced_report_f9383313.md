# Bug Report: pandas.core.indexers.length_of_indexer Incorrect Range Length Calculation

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns incorrect lengths for range objects, including negative values for empty ranges and incorrect counts for ranges with step > 1, violating the invariant that `length_of_indexer(indexer) == len(indexer)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_length_of_indexer_range_consistency(start, stop, step):
    rng = range(start, stop, step)
    expected_length = len(rng)
    predicted_length = length_of_indexer(rng)

    assert expected_length == predicted_length, \
        f"For range({start}, {stop}, {step}): expected {expected_length}, got {predicted_length}"

if __name__ == "__main__":
    test_length_of_indexer_range_consistency()
```

<details>

<summary>
**Failing input**: `range(1, 0, 1)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 18, in <module>
    test_length_of_indexer_range_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 5, in test_length_of_indexer_range_consistency
    start=st.integers(min_value=0, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 14, in test_length_of_indexer_range_consistency
    assert expected_length == predicted_length, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: For range(1, 0, 1): expected 0, got -1
Falsifying example: test_length_of_indexer_range_consistency(
    start=1,
    stop=0,
    step=1,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/3/hypo.py:15
```
</details>

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

# Test case 1: Empty range where start > stop (should be 0, not negative)
rng1 = range(1, 0, 1)
expected_length1 = len(rng1)
predicted_length1 = length_of_indexer(rng1)

print(f"Test case 1: range(1, 0, 1)")
print(f"  Expected length: {expected_length1}")
print(f"  Predicted length: {predicted_length1}")
print(f"  Match: {expected_length1 == predicted_length1}")
print()

# Test case 2: Another empty range
rng2 = range(5, 3, 1)
expected_length2 = len(rng2)
predicted_length2 = length_of_indexer(rng2)

print(f"Test case 2: range(5, 3, 1)")
print(f"  Expected length: {expected_length2}")
print(f"  Predicted length: {predicted_length2}")
print(f"  Match: {expected_length2 == predicted_length2}")
print()

# Test case 3: Non-empty range with step > 1
rng3 = range(0, 5, 2)
expected_length3 = len(rng3)
predicted_length3 = length_of_indexer(rng3)

print(f"Test case 3: range(0, 5, 2)")
print(f"  Expected length: {expected_length3}")
print(f"  Predicted length: {predicted_length3}")
print(f"  Match: {expected_length3 == predicted_length3}")
print()

# Test case 4: Normal range with step = 1
rng4 = range(0, 5, 1)
expected_length4 = len(rng4)
predicted_length4 = length_of_indexer(rng4)

print(f"Test case 4: range(0, 5, 1)")
print(f"  Expected length: {expected_length4}")
print(f"  Predicted length: {predicted_length4}")
print(f"  Match: {expected_length4 == predicted_length4}")
```

<details>

<summary>
Output showing incorrect calculations for multiple range types
</summary>
```
Test case 1: range(1, 0, 1)
  Expected length: 0
  Predicted length: -1
  Match: False

Test case 2: range(5, 3, 1)
  Expected length: 0
  Predicted length: -2
  Match: False

Test case 3: range(0, 5, 2)
  Expected length: 3
  Predicted length: 2
  Match: False

Test case 4: range(0, 5, 1)
  Expected length: 5
  Predicted length: 5
  Match: True
```
</details>

## Why This Is A Bug

The function violates its documented contract to "Return the expected length of target[indexer]" by returning mathematically impossible negative lengths and incorrect positive lengths. The implementation at lines 325-326 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py` uses the incorrect formula:

```python
elif isinstance(indexer, range):
    return (indexer.stop - indexer.start) // indexer.step
```

This formula fails in two critical ways:

1. **Returns negative lengths for empty ranges**: When `start > stop` with positive step (e.g., `range(1, 0, 1)`), the formula yields `(0 - 1) // 1 = -1`. Python's `len(range(1, 0, 1))` correctly returns `0`.

2. **Incorrect calculation for non-unit steps**: For `range(0, 5, 2)`, the formula gives `(5 - 0) // 2 = 2`, but the actual length is `3` (values: 0, 2, 4).

The function must match Python's established range semantics where lengths are always non-negative and calculated using the formula: `max(0, (stop - start + step - 1) // step)` for positive steps.

## Relevant Context

- **Function Location**: `pandas/core/indexers/utils.py:290-329` (specifically lines 325-326)
- **API Status**: Internal/private function in `pandas.core` module
- **Python Range Specification**: Python's [range documentation](https://docs.python.org/3/library/stdtypes.html#range) defines range length calculation
- **Impact**: While not directly exposed to users, this function is used internally by pandas for indexing operations, potentially causing incorrect behavior in downstream operations
- **Test Coverage**: The existing tests likely don't cover edge cases with empty ranges or non-unit steps

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -323,7 +323,10 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        # Calculate range length using Python's formula
+        # For positive step: max(0, (stop - start + step - 1) // step)
+        # This handles empty ranges and non-unit steps correctly
+        return len(indexer)
     elif not is_list_like_indexer(indexer):
         return 1
     raise AssertionError("cannot find the length of the indexer")
```