# Bug Report: pandas.arrays.IntervalArray.overlaps NotImplementedError

**Target**: `pandas.arrays.IntervalArray.overlaps`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`IntervalArray.overlaps()` raises `NotImplementedError` when passed another `IntervalArray`, despite the documentation explicitly stating it accepts `IntervalArray` as a parameter.

## Property-Based Test

```python
import pandas as pd
import pandas.arrays as pa
from hypothesis import given, settings, strategies as st


def interval_strategy(min_val=-100, max_val=100):
    return st.tuples(
        st.floats(allow_nan=False, allow_infinity=False, min_value=min_val, max_value=max_val),
        st.floats(allow_nan=False, allow_infinity=False, min_value=min_val, max_value=max_val)
    ).filter(lambda t: t[0] <= t[1])


@given(st.lists(interval_strategy(), min_size=1, max_size=20))
@settings(max_examples=500)
def test_intervalarray_overlaps_reflexive(intervals):
    arr = pa.IntervalArray.from_tuples(intervals)
    result = arr.overlaps(arr)
    for i, (interval, overlaps_self) in enumerate(zip(arr, result)):
        if pd.notna(interval):
            assert overlaps_self
```

**Failing input**: `intervals=[(0.0, 0.0)]`

## Reproducing the Bug

```python
import pandas as pd
import pandas.arrays as pa

arr = pa.IntervalArray.from_tuples([(0, 1), (2, 3)])
result = arr.overlaps(arr)
```

**Output**:
```
NotImplementedError
```

## Why This Is A Bug

The `overlaps` method documentation explicitly states:

```
Parameters
----------
other : IntervalArray
    Interval to check against for an overlap.
```

The type signature in the docstring promises that `other` should be an `IntervalArray`, but when you actually pass an `IntervalArray`, it raises `NotImplementedError`. This violates the documented contract.

The method works correctly with scalar `Interval` objects, but not with `IntervalArray` objects as documented.

## Fix

The implementation should either:
1. Implement the documented functionality to accept `IntervalArray` inputs
2. Update the documentation to reflect that only scalar `Interval` objects are accepted

Based on the docstring and the natural use case (checking if each interval in one array overlaps with corresponding intervals in another array), option 1 is the correct fix. The implementation needs to be completed to handle `IntervalArray` inputs as documented.