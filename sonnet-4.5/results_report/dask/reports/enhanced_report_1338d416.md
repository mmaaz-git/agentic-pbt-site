# Bug Report: sorted_division_locations Docstring Examples Fail with TypeError

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function fails with `TypeError: No dispatch for <class 'list'>` when called with plain Python lists, despite all 5 examples in its docstring showing lists as input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1).map(sorted),
    chunksize=st.integers(min_value=1, max_value=100)
)
def test_accepts_lists_as_documented(seq, chunksize):
    """Test that sorted_division_locations accepts lists as shown in its examples"""
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert len(divisions) == len(locations)

# Run the test
if __name__ == "__main__":
    test_accepts_lists_as_documented()
```

<details>

<summary>
**Failing input**: `seq=[0], chunksize=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 15, in <module>
    test_accepts_lists_as_documented()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 5, in test_accepts_lists_as_documented
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1).map(sorted),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 10, in test_accepts_lists_as_documented
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
                           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py", line 284, in sorted_division_locations
    seq = tolist(seq)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dispatch.py", line 91, in tolist
    func = tolist_dispatch.dispatch(type(obj))
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 774, in dispatch
    raise TypeError(f"No dispatch for {cls}")
TypeError: No dispatch for <class 'list'>
Falsifying example: test_accepts_lists_as_documented(
    # The test always failed when commented parts were varied together.
    seq=[0],  # or any other generated value
    chunksize=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

# Example directly from the docstring
L = ['A', 'B', 'C', 'D', 'E', 'F']
try:
    divisions, locations = sorted_division_locations(L, chunksize=2)
    print(f"Success: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
Error: TypeError: No dispatch for <class 'list'>
</summary>
```
Error: TypeError: No dispatch for <class 'list'>
```
</details>

## Why This Is A Bug

The function's docstring (lines 257-278 in dask/dataframe/io/io.py) contains 5 examples, all using plain Python lists as input:

1. `sorted_division_locations(['A', 'B', 'C', 'D', 'E', 'F'], chunksize=2)` - Expected: `(['A', 'C', 'E', 'F'], [0, 2, 4, 6])`
2. `sorted_division_locations(['A', 'B', 'C', 'D', 'E', 'F'], chunksize=3)` - Expected: `(['A', 'D', 'F'], [0, 3, 6])`
3. `sorted_division_locations(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'], chunksize=3)` - Expected: `(['A', 'B', 'C', 'C'], [0, 4, 7, 8])`
4. `sorted_division_locations(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'], chunksize=2)` - Expected: `(['A', 'B', 'C', 'C'], [0, 4, 7, 8])`
5. `sorted_division_locations(['A'], chunksize=2)` - Expected: `(['A', 'A'], [0, 1])`

Every single example fails with the same `TypeError`. The function's implementation at line 284 calls `seq = tolist(seq)`, where `tolist` is a dispatch function that only handles numpy arrays, pandas Series/Index/Categorical, and cupy arrays - not plain Python lists. This violates the contract established by the docstring examples.

## Relevant Context

The `tolist` dispatch function (defined in `dask/dataframe/dispatch.py:90-92`) is registered for specific types in `dask/dataframe/backends.py:776`:
- `np.ndarray`
- `pd.Series`
- `pd.Index`
- `pd.Categorical`
- `cupy.ndarray` (if cupy is installed)

Plain Python lists are not registered, causing the dispatch to fail. The comment at line 282-283 says "Convert from an ndarray to a plain list", indicating the function expects numpy arrays as input, but the docstring shows only list examples.

This is an internal utility function in the dask.dataframe.io module, not part of the public API. However, the detailed docstring with examples suggests it may be used by developers extending dask or debugging issues.

## Proposed Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if isinstance(seq, list):
+        seq = list(seq)  # Make a copy for safety
+    else:
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```