# Bug Report: sorted_division_locations Undocumented Precondition

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function has an undocumented precondition that the input sequence must be sorted, but it does not validate this requirement. When called with unsorted input, it returns unsorted divisions, which violates the expected behavior implied by the function name and docstring.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1),
    chunksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=200)
def test_sorted_division_locations_basic_invariants_chunksize(seq, chunksize):
    seq = pd.Series(seq)
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)

    assert divisions == sorted(divisions), "Divisions should be sorted"
```

**Failing input**: `seq=[0, -1], chunksize=1`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

seq = pd.Series([0, -1])
divisions, locations = sorted_division_locations(seq, chunksize=1)

print(f"Input: {list(seq)}")
print(f"Divisions: {divisions}")
print(f"Expected divisions to be sorted: [-1, 0]")
print(f"Actual divisions: {divisions}")
```

Output:
```
Input: [0, -1]
Divisions: [0, -1]
Expected divisions to be sorted: [-1, 0]
Actual divisions: [0, -1]
```

## Why This Is A Bug

1. **Misleading function name**: The function is named `sorted_division_locations`, implying that the divisions should be sorted.

2. **Ambiguous docstring**: The docstring says "Find division locations and values in sorted list" but doesn't clearly state that the input must be sorted.

3. **No validation**: The function doesn't validate that the input is sorted, leading to silent failures.

4. **Caller assumptions**: In `dask/dataframe/dask_expr/io/io.py`, the function is only called when data is known to be sorted:
   ```python
   elif sort or self.frame._data.index.is_monotonic_increasing:
       divisions, locations = sorted_division_locations(...)
   ```
   This confirms the precondition exists, but it's not enforced at the function level.

5. **Contract violation**: The function has an implicit contract that is not documented or validated, making it prone to misuse.

## Fix

The function should validate its precondition. Add a check at the beginning of the function:

```diff
def sorted_division_locations(seq, npartitions=None, chunksize=None):
    """Find division locations and values in sorted list
+
+   Parameters
+   ----------
+   seq : Series or Array
+       Input sequence. Must be sorted in ascending order.
    ...
    """
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")

    seq = tolist(seq)
+   # Validate that seq is sorted
+   if len(seq) > 1 and not all(seq[i] <= seq[i+1] for i in range(len(seq)-1)):
+       raise ValueError("Input sequence must be sorted in ascending order")
+
    # we use bisect later, so we need sorted.
    seq_unique = sorted(set(seq))
    ...
```

Alternatively, the function could automatically sort the input, but this would change the semantics of the `locations` return value.