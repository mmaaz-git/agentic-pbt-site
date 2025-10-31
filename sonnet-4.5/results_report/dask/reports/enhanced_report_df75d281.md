# Bug Report: dask.dataframe.io.io.sorted_division_locations Rejects Plain Python Lists Despite Documented Support

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function raises `TypeError: No dispatch for <class 'list'>` when given plain Python lists, despite its docstring explicitly demonstrating list inputs in all examples.

## Property-Based Test

```python
"""Property-based test for sorted_division_locations with plain Python lists."""
from hypothesis import given, strategies as st, settings
from hypothesis import reproduce_failure
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
def test_sorted_division_locations_accepts_lists(seq, npartitions):
    """Test that sorted_division_locations accepts plain Python lists as documented."""
    seq_sorted = sorted(seq)
    try:
        divisions, locations = sorted_division_locations(seq_sorted, npartitions=npartitions)
        assert len(divisions) == len(locations)
        print(f"✓ Passed with seq={seq_sorted[:5]}{'...' if len(seq_sorted) > 5 else ''}, npartitions={npartitions}")
    except TypeError as e:
        if "No dispatch for <class 'list'>" in str(e):
            print(f"✗ Failed with seq={seq_sorted[:5]}{'...' if len(seq_sorted) > 5 else ''}, npartitions={npartitions}")
            print(f"  Error: {e}")
            raise
        else:
            raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test for sorted_division_locations with lists...")
    print("=" * 70)
    try:
        test_sorted_division_locations_accepts_lists()
        print("=" * 70)
        print("All tests passed!")
    except Exception as e:
        print("=" * 70)
        print(f"Test failed with minimal example!")
        print(f"To reproduce, add this to your test:")
        # The test will fail on the first example, which will be the minimal case
```

<details>

<summary>
**Failing input**: `seq=[0], npartitions=1`
</summary>
```
Running property-based test for sorted_division_locations with lists...
======================================================================
✗ Failed with seq=[0], npartitions=1
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[32], npartitions=7
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[7], npartitions=7
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[4, 14, 36, 42, 75], npartitions=10
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[14, 36, 36, 42, 75], npartitions=10
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[14, 36, 36, 42, 75], npartitions=14
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[14, 36, 36, 36, 75], npartitions=14
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[14, 14, 36, 36, 75], npartitions=14
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[14, 14, 14, 36, 36], npartitions=14
  Error: No dispatch for <class 'list'>
✗ Failed with seq=[0], npartitions=1
  Error: No dispatch for <class 'list'>
[... truncated - all tests fail with same error ...]
======================================================================
Test failed with minimal example!
To reproduce, add this to your test:
```
</details>

## Reproducing the Bug

```python
"""Minimal reproduction of sorted_division_locations bug with plain lists."""
from dask.dataframe.io.io import sorted_division_locations

# Example 1: Simple list from docstring
L = ['A', 'B', 'C', 'D', 'E', 'F']
print("Testing Example 1: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=2")
try:
    result = sorted_division_locations(L, chunksize=2)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting Example 2: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=3")
try:
    result = sorted_division_locations(L, chunksize=3)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Example 3: List with duplicates from docstring
L2 = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
print("\nTesting Example 3: L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'] with chunksize=3")
try:
    result = sorted_division_locations(L2, chunksize=3)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Minimal example
print("\nTesting Minimal Example: L = ['A'] with chunksize=2")
L3 = ['A']
try:
    result = sorted_division_locations(L3, chunksize=2)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError on all list inputs
</summary>
```
Testing Example 1: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=2
Error: TypeError: No dispatch for <class 'list'>

Testing Example 2: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=3
Error: TypeError: No dispatch for <class 'list'>

Testing Example 3: L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'] with chunksize=3
Error: TypeError: No dispatch for <class 'list'>

Testing Minimal Example: L = ['A'] with chunksize=2
Error: TypeError: No dispatch for <class 'list'>
```
</details>

## Why This Is A Bug

The function's docstring creates an explicit API contract that plain Python lists are accepted inputs. Every single example in the docstring (5 total examples from lines 262-277) uses plain Python lists like `['A', 'B', 'C', 'D', 'E', 'F']`. These aren't presented as theoretical - they show specific expected outputs:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

The bug occurs at line 284 where `tolist(seq)` is called. The `tolist` function uses a dispatch system that only has handlers registered for NumPy arrays, pandas Series/Index/Categorical types (see backends.py:776), but no handler for plain Python lists. When a list is passed, the dispatch fails with `TypeError: No dispatch for <class 'list'>`.

The function's own code shows list support was intended - it checks `if isinstance(seq, list)` at the line just before calling tolist, but this logic is broken because tolist can't handle lists.

## Relevant Context

The function is located in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py` starting at line 256.

The dispatch system for `tolist` is defined in:
- `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dispatch.py` (lines 23, 90-92)
- `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/backends.py` (line 776)

The function works correctly with NumPy arrays and pandas Series:
```python
import numpy as np
import pandas as pd

# These work:
sorted_division_locations(np.array(['A', 'B', 'C']), chunksize=2)  # OK
sorted_division_locations(pd.Series(['A', 'B', 'C']), chunksize=2)  # OK

# This fails:
sorted_division_locations(['A', 'B', 'C'], chunksize=2)  # TypeError
```

## Proposed Fix

Add a dispatch handler for list and tuple types in `dask/dataframe/backends.py`:

```diff
@tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
def tolist_numpy_or_pandas(obj):
    return obj.tolist()


+@tolist_dispatch.register((list, tuple))
+def tolist_list_or_tuple(obj):
+    return list(obj)
+
+
@is_categorical_dtype_dispatch.register(
    (pd.Series, pd.Index, pd.api.extensions.ExtensionDtype, np.dtype)
)
```