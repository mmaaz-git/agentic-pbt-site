# Bug Report: pandas.api.indexers.check_array_indexer rejects empty float arrays

**Target**: `pandas.api.indexers.check_array_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pandas.api.indexers.check_array_indexer` inconsistently rejects empty numpy arrays with float dtype while accepting empty Python lists, creating an API inconsistency where `np.array([])` fails but `[]` succeeds for the same logical operation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst
import numpy as np
from pandas.api import indexers

@given(
    npst.arrays(dtype=np.int64, shape=(5,)),
    st.lists(st.integers(min_value=0, max_value=4), min_size=0, max_size=10)
)
def test_check_array_indexer_basic(arr, indices):
    indices_arr = np.array(indices)
    result = indexers.check_array_indexer(arr, indices_arr)
    assert len(result) == len(indices)

if __name__ == "__main__":
    # Run the test
    test_check_array_indexer_basic()
```

<details>

<summary>
**Failing input**: `arr=array([0, 0, 0, 0, 0])`, `indices=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 17, in <module>
    test_check_array_indexer_basic()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 7, in test_check_array_indexer_basic
    npst.arrays(dtype=np.int64, shape=(5,)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 12, in test_check_array_indexer_basic
    result = indexers.check_array_indexer(arr, indices_arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/utils.py", line 551, in check_array_indexer
    raise IndexError("arrays used as indices must be of integer or boolean type")
IndexError: arrays used as indices must be of integer or boolean type
Falsifying example: test_check_array_indexer_basic(
    arr=array([0, 0, 0, 0, 0]),
    indices=[],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.api import indexers

# Test case 1: Empty numpy float array (default dtype) - FAILS
print("Test 1: Empty numpy float array (default dtype)")
arr = np.array([1, 2, 3, 4, 5])
empty_float_arr = np.array([])  # Default dtype is float64
print(f"  empty_float_arr dtype: {empty_float_arr.dtype}")

try:
    result = indexers.check_array_indexer(arr, empty_float_arr)
    print(f"  Result: {result}")
except IndexError as e:
    print(f"  Error: {e}")

print()

# Test case 2: Empty Python list - WORKS
print("Test 2: Empty Python list")
try:
    result = indexers.check_array_indexer(arr, [])
    print(f"  Result: {result}")
    print(f"  Result dtype: {result.dtype}")
except IndexError as e:
    print(f"  Error: {e}")

print()

# Test case 3: Empty numpy integer array - WORKS
print("Test 3: Empty numpy integer array (explicit dtype)")
empty_int_arr = np.array([], dtype=np.int64)
print(f"  empty_int_arr dtype: {empty_int_arr.dtype}")

try:
    result = indexers.check_array_indexer(arr, empty_int_arr)
    print(f"  Result: {result}")
    print(f"  Result dtype: {result.dtype}")
except IndexError as e:
    print(f"  Error: {e}")

print()

# Test case 4: Empty boolean array - FAILS differently
print("Test 4: Empty numpy boolean array")
empty_bool_arr = np.array([], dtype=bool)
print(f"  empty_bool_arr dtype: {empty_bool_arr.dtype}")

try:
    result = indexers.check_array_indexer(arr, empty_bool_arr)
    print(f"  Result: {result}")
    print(f"  Result dtype: {result.dtype}")
except IndexError as e:
    print(f"  Error: {e}")
```

<details>

<summary>
Error: arrays used as indices must be of integer or boolean type
</summary>
```
Test 1: Empty numpy float array (default dtype)
  empty_float_arr dtype: float64
  Error: arrays used as indices must be of integer or boolean type

Test 2: Empty Python list
  Result: []
  Result dtype: int64

Test 3: Empty numpy integer array (explicit dtype)
  empty_int_arr dtype: int64
  Result: []
  Result dtype: int64

Test 4: Empty numpy boolean array
  empty_bool_arr dtype: bool
  Error: Boolean index has wrong length: 0 instead of 5
```
</details>

## Why This Is A Bug

This bug violates the principle of consistent API behavior. The function `check_array_indexer` has special handling for empty Python lists (lines 526-528 in pandas/core/indexers/utils.py) but fails to apply the same logic to empty numpy arrays:

1. **Inconsistent behavior**: The function checks `is_array_like(indexer)` to determine if the input needs conversion. Empty lists return False (not array-like) and get special handling, but empty numpy arrays return True (array-like) and skip this critical path.

2. **Developer awareness**: Line 527 contains the comment "empty list is converted to float array by pd.array" showing developers knew about the dtype issue for empty collections but only fixed it for one code path.

3. **Logical contradiction**: An empty array has zero elements to validate, making dtype checking meaningless for indexing purposes. Whether dtype is float64, int64, or bool, an empty array indexes nothing.

4. **Documentation mismatch**: While the docstring shows float64 arrays should raise errors, it doesn't specify that empty arrays should be treated differently based on their creation method (`[]` vs `np.array([])`).

## Relevant Context

The bug stems from a control flow issue in the source code (pandas/core/indexers/utils.py):

- Lines 524-528: Handle non-array-like inputs (e.g., Python lists), including special case for empty lists
- Lines 530-551: Handle array-like inputs, checking dtype without considering empty arrays
- The `is_array_like()` function treats `[]` and `np.array([])` differently, causing them to take different code paths

Key source code link: https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexers/utils.py#L524-L551

The fact that `np.array([])` defaults to float64 dtype is a numpy design decision that pandas should handle gracefully, just as it already does for empty Python lists.

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -524,9 +524,13 @@ def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
     if not is_array_like(indexer):
         indexer = pd_array(indexer)
         if len(indexer) == 0:
             # empty list is converted to float array by pd.array
             indexer = np.array([], dtype=np.intp)
+    # Handle empty numpy arrays regardless of dtype
+    elif len(indexer) == 0:
+        # Empty arrays should be treated as integer arrays for indexing
+        indexer = np.array([], dtype=np.intp)

     dtype = indexer.dtype
     if is_bool_dtype(dtype):
```