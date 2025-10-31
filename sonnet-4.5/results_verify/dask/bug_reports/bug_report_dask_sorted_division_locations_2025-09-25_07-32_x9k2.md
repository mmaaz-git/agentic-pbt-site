# Bug Report: sorted_division_locations Silently Produces Invalid Results on Unsorted Input

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function silently produces invalid results (non-monotonic locations) when given unsorted input, instead of validating its precondition or raising an error.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations


@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_sorted_division_locations_monotonic_locations(seq, chunksize):
    seq_pd = pd.Series(seq)
    divisions, locations = sorted_division_locations(seq_pd, chunksize=chunksize)

    for i in range(len(locations) - 1):
        assert locations[i] < locations[i+1], \
            f"Locations must be strictly increasing: locations[{i}]={locations[i]} >= locations[{i+1}]={locations[i+1]}"
```

**Failing input**: `seq=[0, 1, 0, 0], chunksize=1`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

seq = pd.Series([0, 1, 0, 0])
divisions, locations = sorted_division_locations(seq, chunksize=1)

print(f"Input (unsorted): {list(seq)}")
print(f"Divisions: {divisions}")
print(f"Locations: {locations}")
```

**Output:**
```
Input (unsorted): [0, 1, 0, 0]
Divisions: [0, 1, 0]
Locations: [0, 4, 4]
```

**Bug**: `locations[1] = locations[2] = 4`, violating the invariant that locations should be strictly monotonically increasing.

## Why This Is A Bug

The function's docstring states it operates on a "sorted list" and returns division locations. The returned locations should always be strictly monotonically increasing, as they represent indices into the sequence. However, when given unsorted input:

1. The function does not validate that the input is sorted
2. It silently produces invalid results (duplicate locations)
3. This violates the fundamental invariant that `locations[i] < locations[i+1]`

While internal callers properly ensure sorted input, the function is part of the public API and should either:
- Validate its precondition and raise a clear error for unsorted input
- Explicitly document that unsorted input leads to undefined behavior

## Fix

Add input validation at the beginning of the function:

```diff
def sorted_division_locations(seq, npartitions=None, chunksize=None):
    """Find division locations and values in sorted list

+   The input sequence must be sorted. Unsorted input will raise ValueError.
+
    Examples
    --------

    >>> L = ['A', 'B', 'C', 'D', 'E', 'F']
    >>> sorted_division_locations(L, chunksize=2)
    (['A', 'C', 'E', 'F'], [0, 2, 4, 6])
    """
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")

    seq = tolist(seq)
+
+   # Validate that input is sorted
+   if len(seq) > 1:
+       for i in range(len(seq) - 1):
+           if seq[i] > seq[i+1]:
+               raise ValueError(
+                   f"Input sequence must be sorted. Found seq[{i}]={seq[i]} > seq[{i+1}]={seq[i+1]}"
+               )
+
    # we use bisect later, so we need sorted.
    seq_unique = sorted(set(seq))
    duplicates = len(seq_unique) < len(seq)
    enforce_exact = False
    ...
```