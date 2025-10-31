# Bug Report: pandas.core.util.hashing.combine_hash_arrays Bypasses Assertion on Empty Iterator

**Target**: `pandas.core.util.hashing.combine_hash_arrays`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_hash_arrays` function contains an assertion to validate that the number of items processed matches the expected `num_items` parameter. However, this assertion is bypassed when the input iterator is empty, allowing inconsistent state where `num_items > 0` but zero arrays are actually processed.

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
```

**Failing input**: `num_items=1` (or any positive integer)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import combine_hash_arrays

result = combine_hash_arrays(iter([]), 1)
print(f"Result: {result}")
print(f"Expected: AssertionError with message 'Fed in wrong num_items'")
print(f"Actual: Returns empty array without error")

arr = np.array([1, 2, 3], dtype=np.uint64)
try:
    result = combine_hash_arrays(iter([arr]), 2)
except AssertionError as e:
    print(f"\nFor comparison, non-empty case correctly raises: {e}")
```

Output:
```
Result: []
Expected: AssertionError with message 'Fed in wrong num_items'
Actual: Returns empty array without error

For comparison, non-empty case correctly raises: Fed in wrong num_items
```

## Why This Is A Bug

The function has an assertion at line 78 that validates `last_i + 1 == num_items` to catch programming errors where the caller provides an incorrect `num_items` value. However, when the iterator is empty, the function returns early at line 65 without performing this validation.

This creates an inconsistency:
- Non-empty case: `combine_hash_arrays(iter([arr]), 2)` → AssertionError ✓
- Empty case: `combine_hash_arrays(iter([]), 1)` → silently succeeds ✗

The assertion is meant to catch ALL cases where `num_items` doesn't match the actual count, but the early return bypasses this check.

## Fix

Add validation before the early return to ensure `num_items` is 0 when the iterator is empty:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -62,6 +62,8 @@ def combine_hash_arrays(
     try:
         first = next(arrays)
     except StopIteration:
+        if num_items != 0:
+            raise AssertionError(f"Fed in wrong num_items: expected 0, got {num_items}")
         return np.array([], dtype=np.uint64)

     arrays = itertools.chain([first], arrays)
```