# Bug Report: pandas.core.util.hashing.combine_hash_arrays Empty Iterator Validation

**Target**: `pandas.core.util.hashing.combine_hash_arrays`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_hash_arrays` function fails to validate the `num_items` parameter when passed an empty iterator, allowing invalid inputs to silently succeed instead of raising an AssertionError as intended.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import combine_hash_arrays
import pytest


@given(
    st.lists(st.lists(st.integers(min_value=0, max_value=2**32), min_size=1, max_size=5), min_size=0, max_size=20),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_combine_hash_arrays_num_items(hash_lists, num_items):
    arrays = [np.array(h, dtype=np.uint64) for h in hash_lists]
    if len(arrays) == num_items and num_items > 0:
        result = combine_hash_arrays(iter(arrays), num_items)
        assert result.dtype == np.uint64
    elif len(arrays) != num_items:
        with pytest.raises(AssertionError, match="Fed in wrong num_items"):
            combine_hash_arrays(iter(arrays), num_items)
```

**Failing input**: `hash_lists=[], num_items=1`

## Reproducing the Bug

```python
from pandas.core.util.hashing import combine_hash_arrays

result = combine_hash_arrays(iter([]), 1)
print(f"Result: {result}")
print(f"Expected: AssertionError('Fed in wrong num_items')")
print(f"Actual: Returns empty array {result} without error")

result = combine_hash_arrays(iter([]), 5)
print(f"With num_items=5: {result}")
```

## Why This Is A Bug

The function includes an assertion on line 78 to validate that the number of items fed matches the expected `num_items` parameter:

```python
assert last_i + 1 == num_items, "Fed in wrong num_items"
```

However, when the iterator is empty, the for loop (lines 72-76) never executes, so `last_i` remains at its initial value of 0, and the assertion is never reached. This allows invalid calls like `combine_hash_arrays(iter([]), 5)` to silently return an empty array instead of raising an error.

The assertion's purpose is to catch programming errors where the caller miscounts the items, but it fails to do so when the iterator is empty.

## Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -68,13 +68,14 @@ def combine_hash_arrays(

     mult = np.uint64(1000003)
     out = np.zeros_like(first) + np.uint64(0x345678)
-    last_i = 0
+    last_i = -1
     for i, a in enumerate(arrays):
         inverse_i = num_items - i
         out ^= a
         out *= mult
         mult += np.uint64(82520 + inverse_i + inverse_i)
         last_i = i
     assert last_i + 1 == num_items, "Fed in wrong num_items"
     out += np.uint64(97531)
     return out
```

By initializing `last_i = -1` instead of `last_i = 0`, the assertion will correctly catch the case where an empty iterator is passed with `num_items > 0`.