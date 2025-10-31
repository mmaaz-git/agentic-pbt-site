# Bug Report: dask.dataframe.io.io.sorted_division_locations TypeError with List Input

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function fails with a `TypeError` when given a Python list as input, even though the function's own docstring explicitly demonstrates using lists in all of its examples.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_sorted_division_locations_invariants(seq, npartitions):
    assume(npartitions <= len(seq))
    divisions, locations = sorted_division_locations(seq, npartitions=npartitions)
    assert len(divisions) == len(locations)

# Run the test
if __name__ == "__main__":
    test_sorted_division_locations_invariants()
```

<details>

<summary>
**Failing input**: `seq=[0], npartitions=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 16, in <module>
    test_sorted_division_locations_invariants()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 5, in test_sorted_division_locations_invariants
    seq=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 11, in test_sorted_division_locations_invariants
    divisions, locations = sorted_division_locations(seq, npartitions=npartitions)
                           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py", line 284, in sorted_division_locations
    seq = tolist(seq)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dispatch.py", line 91, in tolist
    func = tolist_dispatch.dispatch(type(obj))
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 774, in dispatch
    raise TypeError(f"No dispatch for {cls}")
TypeError: No dispatch for <class 'list'>
Falsifying example: test_sorted_division_locations_invariants(
    seq=[0],  # or any other generated value
    npartitions=1,
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

# Example directly from the function's docstring (line 262-264 of io.py)
# This should work according to the documentation but doesn't
L = ['A', 'B', 'C', 'D', 'E', 'F']
print(f"Input list: {L}")
print(f"Calling sorted_division_locations(L, chunksize=2)")
print(f"Expected (from docstring): (['A', 'C', 'E', 'F'], [0, 2, 4, 6])")
print()

result = sorted_division_locations(L, chunksize=2)
```

<details>

<summary>
TypeError: No dispatch for <class 'list'>
</summary>
```
Input list: ['A', 'B', 'C', 'D', 'E', 'F']
Calling sorted_division_locations(L, chunksize=2)
Expected (from docstring): (['A', 'C', 'E', 'F'], [0, 2, 4, 6])

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/repo.py", line 11, in <module>
    result = sorted_division_locations(L, chunksize=2)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py", line 284, in sorted_division_locations
    seq = tolist(seq)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dispatch.py", line 91, in tolist
    func = tolist_dispatch.dispatch(type(obj))
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 774, in dispatch
    raise TypeError(f"No dispatch for {cls}")
TypeError: No dispatch for <class 'list'>
```
</details>

## Why This Is A Bug

The function's docstring contains six examples (lines 262-277 in io.py), and every single one uses a Python list as the input parameter. These doctest-style examples establish a clear contract that lists are valid inputs. However, when any of these documented examples are executed, they fail with a `TypeError`.

The root cause is in the dispatcher system. At line 284, the function calls `tolist(seq)` which uses a dispatch mechanism (`tolist_dispatch`) to convert various array-like objects to lists. The dispatcher has registered handlers for numpy arrays, pandas objects, and cupy arrays (found in `dask/dataframe/backends.py`), but crucially lacks a handler for Python's built-in list type. When a list is passed, the dispatcher cannot find an appropriate handler and raises `TypeError: No dispatch for <class 'list'>`.

This is particularly ironic because the function needs to convert inputs to lists for its internal operations (it uses `bisect` and list operations throughout), yet it cannot accept lists as input. The comment at lines 282-283 states "Convert from an ndarray to a plain list" but the implementation blindly attempts conversion even when the input is already a list.

## Relevant Context

The `sorted_division_locations` function is used internally by dask to determine how to partition sorted data. While most users interact with it indirectly through higher-level DataFrame operations (which typically work with numpy/pandas objects), the function is part of the public API and its documentation explicitly shows list usage.

The dispatcher pattern is found in:
- `dask/dataframe/dispatch.py`: Defines `tolist` function and `tolist_dispatch`
- `dask/dataframe/backends.py`: Contains registrations for numpy/pandas/cupy types
- Current registrations at line 776: `@tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))`

The issue has likely gone unnoticed because:
1. Most real-world usage comes through DataFrame operations that already use numpy/pandas types
2. The workaround is simple: convert lists to numpy arrays first
3. The function still works correctly for its primary use cases

## Proposed Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -279,9 +279,11 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):
     if (npartitions is None) == (chunksize is None):
         raise ValueError("Exactly one of npartitions and chunksize must be specified.")

-    # Convert from an ndarray to a plain list so that
-    # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    # Convert from an ndarray (or other array-like) to a plain list so that
+    # any divisions we extract from seq are plain Python scalars.
+    # Skip conversion if already a list to avoid dispatcher error.
+    if not isinstance(seq, list):
+        seq = tolist(seq)
+
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```