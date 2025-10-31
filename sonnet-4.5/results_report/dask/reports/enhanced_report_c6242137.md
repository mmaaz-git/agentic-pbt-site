# Bug Report: dask.dataframe.io.sorted_division_locations TypeError with Plain Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function crashes with `TypeError: No dispatch for <class 'list'>` when called with plain Python lists, despite its docstring explicitly showing multiple examples using plain lists as input.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations


@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=1000)
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]

# Run the test
if __name__ == "__main__":
    test_sorted_division_locations_accepts_lists()
```

<details>

<summary>
**Failing input**: `seq=[0], chunksize=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 18, in <module>
    test_sorted_division_locations_accepts_lists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 6, in test_sorted_division_locations_accepts_lists
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 12, in test_sorted_division_locations_accepts_lists
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
                           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py", line 284, in sorted_division_locations
    seq = tolist(seq)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dispatch.py", line 91, in tolist
    func = tolist_dispatch.dispatch(type(obj))
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 774, in dispatch
    raise TypeError(f"No dispatch for {cls}")
TypeError: No dispatch for <class 'list'>
Falsifying example: test_sorted_division_locations_accepts_lists(
    seq=[0],
    chunksize=1,
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

# Example directly from the docstring
L = ['A', 'B', 'C', 'D', 'E', 'F']
print(f"Input list: {L}")
print(f"Calling sorted_division_locations(L, chunksize=2)")

try:
    divisions, locations = sorted_division_locations(L, chunksize=2)
    print(f"Success! divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError when calling with plain list
</summary>
```
Input list: ['A', 'B', 'C', 'D', 'E', 'F']
Calling sorted_division_locations(L, chunksize=2)
Error: TypeError: No dispatch for <class 'list'>
```
</details>

## Why This Is A Bug

The function's docstring contains three examples (lines 262-277 of io.py) that all use plain Python lists as input:

1. `L = ['A', 'B', 'C', 'D', 'E', 'F']` with expected output `(['A', 'C', 'E', 'F'], [0, 2, 4, 6])`
2. `L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']` demonstrating handling of duplicates
3. `L = ['A']` demonstrating single-element list handling

However, the implementation on line 284 calls `seq = tolist(seq)` unconditionally. The `tolist()` function uses a dispatch mechanism (`tolist_dispatch`) that is only registered for `(np.ndarray, pd.Series, pd.Index, pd.Categorical)` types as seen in `dask/dataframe/backends.py:776`. When a plain Python list is passed, the dispatcher raises `TypeError: No dispatch for <class 'list'>`.

This creates a contradiction where the function's documentation shows it accepting lists, but the implementation crashes on exactly those inputs. The comment on lines 282-283 states "Convert from an ndarray to a plain list so that any divisions we extract from seq are plain Python scalars," which suggests the function expects non-list inputs but wants to work with lists internally.

## Relevant Context

The `tolist()` function is defined in `dask/dataframe/dispatch.py:90-92`:
```python
def tolist(obj):
    func = tolist_dispatch.dispatch(type(obj))
    return func(obj)
```

The dispatch registration in `dask/dataframe/backends.py:776-778` shows:
```python
@tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
def tolist_numpy_or_pandas(obj):
    return obj.tolist()
```

This explains why lists fail - there's no dispatch handler for the `list` type itself. The function assumes all inputs need conversion to lists, but doesn't handle the case where the input is already a list.

The issue appears to be fixed in newer versions of dask (seen in `dask_env` directory) but exists in version 2025.9.1.

## Proposed Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if isinstance(seq, list):
+        pass
+    else:
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```