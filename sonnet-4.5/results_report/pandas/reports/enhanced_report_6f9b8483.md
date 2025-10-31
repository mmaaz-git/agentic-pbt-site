# Bug Report: pandas.core.util.hashing.hash_tuples Empty List Crash

**Target**: `pandas.core.util.hashing.hash_tuples`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `hash_tuples()` function crashes with a `TypeError` when given an empty list input, while the related `hash_array()` function handles empty inputs gracefully by returning an empty uint64 array.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_tuples
import numpy as np
import pytest


def test_hash_tuples_empty():
    """Test that hash_tuples handles empty list input gracefully."""
    hashed = hash_tuples([])
    assert len(hashed) == 0
    assert hashed.dtype == np.uint64


if __name__ == "__main__":
    test_hash_tuples_empty()
```

<details>

<summary>
**Failing input**: `[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo_hashtuples.py", line 15, in <module>
    test_hash_tuples_empty()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo_hashtuples.py", line 9, in test_hash_tuples_empty
    hashed = hash_tuples([])
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/util/hashing.py", line 210, in hash_tuples
    mi = MultiIndex.from_tuples(vals)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/multi.py", line 223, in new_meth
    return meth(self_or_cls, *args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/multi.py", line 610, in from_tuples
    raise TypeError("Cannot infer number of levels from empty list")
TypeError: Cannot infer number of levels from empty list
```
</details>

## Reproducing the Bug

```python
from pandas.core.util.hashing import hash_tuples, hash_array
import numpy as np

# First show that hash_array handles empty input gracefully
empty_arr = np.array([], dtype=np.int64)
hash_arr_result = hash_array(empty_arr)
print(f"hash_array([]) works: {hash_arr_result}")
print(f"hash_array([]) dtype: {hash_arr_result.dtype}")

# Now show that hash_tuples crashes with empty input
print("\nCalling hash_tuples([])...")
try:
    result = hash_tuples([])
    print(f"hash_tuples([]) result: {result}")
except Exception as e:
    print(f"hash_tuples([]) raised {e.__class__.__name__}: {e}")
```

<details>

<summary>
TypeError raised when hash_tuples called with empty list
</summary>
```
hash_array([]) works: []
hash_array([]) dtype: uint64

Calling hash_tuples([])...
hash_tuples([]) raised TypeError: Cannot infer number of levels from empty list
```
</details>

## Why This Is A Bug

This is a bug because it violates several key principles:

1. **API Inconsistency**: The `hash_array()` function in the same module (`pandas.core.util.hashing`) handles empty inputs gracefully by returning an empty uint64 array, but `hash_tuples()` crashes. Users would reasonably expect consistent behavior between these closely related functions.

2. **Type Contract Violation**: The function signature accepts `Iterable[tuple[Hashable, ...]]` according to the type hints, which includes empty iterables. The function accepts any "listlike-of-tuples" according to its docstring, which should include empty lists.

3. **Missing Documentation**: The docstring for `hash_tuples()` does not specify that empty lists are invalid input. It simply states it hashes "MultiIndex / listlike-of-tuples efficiently" without mentioning any restrictions on empty inputs.

4. **Implementation Detail Leakage**: The error occurs at line 210 where `MultiIndex.from_tuples(vals)` is called. This is an internal implementation detail that should not dictate the public API behavior, especially when the underlying `combine_hash_arrays` function (lines 62-65 in hashing.py) already handles empty iterators correctly.

5. **Principle of Least Surprise**: Empty collections are standard edge cases in programming. Most pandas/NumPy functions handle empty inputs by returning appropriately typed empty outputs. Users would expect `hash_tuples([])` to return `np.array([], dtype=np.uint64)`.

## Relevant Context

The issue stems from the implementation at line 210 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/util/hashing.py`:

```python
if not isinstance(vals, ABCMultiIndex):
    mi = MultiIndex.from_tuples(vals)  # This raises TypeError for empty lists
```

However, the infrastructure to handle empty inputs already exists in the same file. The `combine_hash_arrays` function (lines 62-65) explicitly handles empty iterators:

```python
try:
    first = next(arrays)
except StopIteration:
    return np.array([], dtype=np.uint64)
```

This shows that empty handling was considered for the internal functions but not exposed consistently in the public API.

Documentation reference: The [pandas.util.hash_pandas_object documentation](https://pandas.pydata.org/docs/reference/api/pandas.util.hash_pandas_object.html) and related hashing utilities don't specify restrictions on empty inputs.

## Proposed Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -200,6 +200,10 @@ def hash_tuples(
     """
     if not is_list_like(vals):
         raise TypeError("must be convertible to a list-of-tuples")
+
+    # Handle empty input consistently with hash_array
+    if len(list(vals)) == 0:
+        return np.array([], dtype=np.uint64)

     from pandas import (
         Categorical,
```