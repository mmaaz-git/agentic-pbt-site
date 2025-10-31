# Bug Report: pandas.core.util.hashing.combine_hash_arrays Bypasses Assertion Check for Empty Iterators

**Target**: `pandas.core.util.hashing.combine_hash_arrays`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_hash_arrays` function contains an assertion that validates the number of arrays processed matches the `num_items` parameter, but this validation is bypassed when the input iterator is empty, allowing inconsistent states where `num_items > 0` with zero arrays processed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.util.hashing import combine_hash_arrays
import pytest

@given(st.integers(min_value=1, max_value=10))
def test_combine_hash_arrays_empty_with_nonzero_count(num_items):
    arrays = iter([])
    # Should raise AssertionError since num_items > 0 but no arrays provided
    # but it silently succeeds instead
    result = combine_hash_arrays(arrays, num_items)
    # This should not be reached without an error
    assert False, f"Expected assertion error for num_items={num_items} with empty iterator"

if __name__ == "__main__":
    test_combine_hash_arrays_empty_with_nonzero_count()
```

<details>

<summary>
**Failing input**: `num_items=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 15, in <module>
    test_combine_hash_arrays_empty_with_nonzero_count()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 6, in test_combine_hash_arrays_empty_with_nonzero_count
    def test_combine_hash_arrays_empty_with_nonzero_count(num_items):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 12, in test_combine_hash_arrays_empty_with_nonzero_count
    assert False, f"Expected assertion error for num_items={num_items} with empty iterator"
           ^^^^^
AssertionError: Expected assertion error for num_items=1 with empty iterator
Falsifying example: test_combine_hash_arrays_empty_with_nonzero_count(
    num_items=1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import combine_hash_arrays

# Test 1: Empty iterator with num_items=1 (should raise AssertionError but doesn't)
print("Test 1: Empty iterator with num_items=1")
result = combine_hash_arrays(iter([]), 1)
print(f"Result: {result}")
print(f"Result type: {type(result)}")
print(f"Result shape: {result.shape}")
print(f"Expected: AssertionError with message 'Fed in wrong num_items'")
print(f"Actual: Returns empty array without error")
print()

# Test 2: Empty iterator with various num_items values
print("Test 2: Empty iterator with various num_items values")
for num_items in [2, 5, 10]:
    result = combine_hash_arrays(iter([]), num_items)
    print(f"num_items={num_items}: result={result}, shape={result.shape}")
print()

# Test 3: Empty iterator with num_items=0 (correct case)
print("Test 3: Empty iterator with num_items=0 (should succeed)")
result = combine_hash_arrays(iter([]), 0)
print(f"Result: {result}")
print(f"This correctly returns an empty array")
print()

# Test 4: Non-empty case with wrong num_items (for comparison)
print("Test 4: Non-empty iterator with wrong num_items (should raise AssertionError)")
arr = np.array([1, 2, 3], dtype=np.uint64)
try:
    result = combine_hash_arrays(iter([arr]), 2)
    print(f"Unexpectedly succeeded: {result}")
except AssertionError as e:
    print(f"Correctly raised AssertionError: {e}")
print()

# Test 5: Non-empty case with correct num_items (should succeed)
print("Test 5: Non-empty iterator with correct num_items (should succeed)")
arr = np.array([1, 2, 3], dtype=np.uint64)
result = combine_hash_arrays(iter([arr]), 1)
print(f"Result: {result}")
print(f"This correctly processes the array")
```

<details>

<summary>
Empty iterator with num_items=1 returns empty array instead of raising AssertionError
</summary>
```
Test 1: Empty iterator with num_items=1
Result: []
Result type: <class 'numpy.ndarray'>
Result shape: (0,)
Expected: AssertionError with message 'Fed in wrong num_items'
Actual: Returns empty array without error

Test 2: Empty iterator with various num_items values
num_items=2: result=[], shape=(0,)
num_items=5: result=[], shape=(0,)
num_items=10: result=[], shape=(0,)

Test 3: Empty iterator with num_items=0 (should succeed)
Result: []
This correctly returns an empty array

Test 4: Non-empty iterator with wrong num_items (should raise AssertionError)
Correctly raised AssertionError: Fed in wrong num_items

Test 5: Non-empty iterator with correct num_items (should succeed)
Result: [3430019387558 3430020387561 3430021387564]
This correctly processes the array
```
</details>

## Why This Is A Bug

The function contains an assertion at line 78 (`assert last_i + 1 == num_items, "Fed in wrong num_items"`) designed to catch programming errors where the caller provides an incorrect `num_items` value. This assertion correctly validates non-empty iterators but is bypassed for empty iterators due to an early return at line 65.

This creates inconsistent behavior:
- When a non-empty iterator is provided with wrong `num_items` (e.g., `combine_hash_arrays(iter([arr]), 2)`), the assertion correctly raises an error
- When an empty iterator is provided with wrong `num_items` (e.g., `combine_hash_arrays(iter([]), 1)`), the function silently returns an empty array without validation

The assertion's purpose is to ensure the caller's expectation (`num_items`) matches reality (actual array count), serving as a consistency check to catch programming errors. By bypassing this check for empty iterators, the function fails to enforce its contract consistently.

## Relevant Context

The `combine_hash_arrays` function is an internal utility in `pandas.core.util.hashing` used for combining hash values from multiple arrays. It's called internally by:
- `hash_pandas_object` at line 148 for Series with index (always with `num_items=2`)
- `hash_pandas_object` at line 174 for DataFrames (with `num_items` calculated from column count)
- `hash_tuples` at line 228 for MultiIndex (with `num_items=len(cat_vals)`)

All internal callers correctly provide the exact count, so this bug doesn't affect pandas' normal operation. However, the inconsistent assertion behavior could mask programming errors during development or testing.

The function documentation references CPython's tupleobject.c, suggesting it should behave similarly to Python's tuple hashing. However, CPython doesn't have a separate `num_items` parameter - it determines the count from the tuple itself, making this validation unnecessary there.

## Proposed Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -62,6 +62,8 @@ def combine_hash_arrays(
     try:
         first = next(arrays)
     except StopIteration:
+        if num_items != 0:
+            raise AssertionError(f"Fed in wrong num_items")
         return np.array([], dtype=np.uint64)

     arrays = itertools.chain([first], arrays)
```