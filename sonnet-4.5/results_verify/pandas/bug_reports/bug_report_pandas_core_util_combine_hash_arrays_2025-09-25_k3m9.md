# Bug Report: pandas.core.util.hashing.combine_hash_arrays Bypasses Assertion on Empty Iterator

**Target**: `pandas.core.util.hashing.combine_hash_arrays`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `combine_hash_arrays` function contains an assertion to validate that the number of arrays provided matches the `num_items` parameter. However, this assertion is bypassed when an empty iterator is provided, allowing invalid inputs to succeed silently.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import combine_hash_arrays


@given(st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_combine_hash_arrays_empty_iterator_assertion(num_items):
    arrays = iter([])
    try:
        result = combine_hash_arrays(arrays, num_items)
        assert False, f"Expected AssertionError but got result: {result}"
    except AssertionError as e:
        if "Fed in wrong num_items" not in str(e):
            raise
```

**Failing input**: `num_items=5` (or any positive integer)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import combine_hash_arrays

result = combine_hash_arrays(iter([]), 5)
print(f"Result: {result}")
print(f"Expected: AssertionError('Fed in wrong num_items')")
print(f"Actual: Returned empty array {result}")
```

## Why This Is A Bug

The function includes an explicit assertion (`assert last_i + 1 == num_items, "Fed in wrong num_items"`) to validate that the actual number of arrays matches the claimed `num_items` parameter. This contract is violated when the iterator is empty and `num_items > 0`, because:

1. The function returns early on line 65 when catching `StopIteration`
2. This bypasses the assertion on line 78 that validates the count
3. The function returns successfully even though the input violates its contract

The docstring states that `num_items` should match the actual number of items in the iterator. When they don't match, the function should fail with an assertion error, but it doesn't when the iterator is empty.

## Fix

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -62,6 +62,8 @@ def combine_hash_arrays(
     try:
         first = next(arrays)
     except StopIteration:
+        if num_items != 0:
+            raise AssertionError("Fed in wrong num_items")
         return np.array([], dtype=np.uint64)

     arrays = itertools.chain([first], arrays)
```