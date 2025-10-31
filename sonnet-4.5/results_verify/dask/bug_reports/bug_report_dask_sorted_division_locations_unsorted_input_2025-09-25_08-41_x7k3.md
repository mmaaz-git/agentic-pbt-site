# Bug Report: sorted_division_locations Returns Unsorted Divisions

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The function `sorted_division_locations` returns unsorted division values when given an unsorted input sequence, violating its documented contract and breaking code that depends on sorted divisions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1).map(pd.Series),
    chunksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_sorted_division_locations_divisions_sorted(seq, chunksize):
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert divisions == sorted(divisions)
```

**Failing input**: `seq=pd.Series([0, -1]), chunksize=1`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

seq = pd.Series([0, -1])
divisions, locations = sorted_division_locations(seq, chunksize=1)

print(f"Input: {seq.tolist()}")
print(f"Divisions: {divisions}")
print(f"Expected: {sorted(divisions)}")

assert divisions == sorted(divisions)
```

Output:
```
Input: [0, -1]
Divisions: [0, -1]
Expected: [-1, 0]
AssertionError
```

## Why This Is A Bug

1. **Function name implies sorted output**: The function is called `sorted_division_locations`, suggesting it should return sorted divisions
2. **Docstring says "sorted list"**: The docstring explicitly states "Find division locations and values in sorted list"
3. **All examples use sorted input**: Every example in the docstring uses already-sorted sequences, hiding this bug
4. **Code depends on sorted data**: The implementation uses `bisect` operations which require sorted data (line 294: `bisect.bisect_left(seq, x)`)

The function has an **undocumented precondition** that the input must be sorted, but it neither validates this nor documents it. This violates the principle of least surprise and can cause silent data corruption in downstream code that expects sorted divisions.

## Fix

The function should either:
1. **Sort the input** before processing, OR
2. **Validate and raise an error** if the input is not sorted

Option 1 (sort the input) is safer and more user-friendly:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -283,6 +283,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):
     if isinstance(seq, list):
         pass
     else:
         seq = tolist(seq)
+
+    # Ensure sequence is sorted (required for bisect operations and division semantics)
+    seq = sorted(seq)
+
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
```