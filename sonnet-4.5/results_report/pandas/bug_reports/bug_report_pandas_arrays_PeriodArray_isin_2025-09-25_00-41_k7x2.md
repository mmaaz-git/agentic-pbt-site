# Bug Report: PeriodArray.isin Crashes with List Input

**Target**: `pandas.arrays.PeriodArray.isin`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`PeriodArray.isin()` crashes with `AttributeError` when given a list as input, while other pandas ExtensionArrays (like `IntegerArray`) correctly accept lists. This is an API inconsistency.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st


@st.composite
def period_arrays(draw, min_size=1, max_size=50):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    freq = draw(st.sampled_from(['D', 'M', 'Y']))

    years = draw(st.lists(st.integers(min_value=2000, max_value=2030), min_size=size, max_size=size))
    months = draw(st.lists(st.integers(min_value=1, max_value=12), min_size=size, max_size=size))

    periods = [pd.Period(f'{y}-{m:02d}', freq=freq) for y, m in zip(years, months)]
    return pd.array(periods, dtype=f'period[{freq}]')


@given(period_arrays(min_size=1))
def test_period_array_isin_consistency(arr):
    first_val = None
    for val in arr:
        if pd.notna(val):
            first_val = val
            break

    if first_val is not None:
        result = arr.isin([first_val])

        for i, val in enumerate(arr):
            if pd.notna(val) and val == first_val:
                assert result[i]
```

**Failing input**: `arr = PeriodArray(['2000-01-01'], dtype='period[D]')`

## Reproducing the Bug

```python
import pandas as pd

period = pd.Period('2000-01', freq='D')
arr = pd.array([period], dtype='period[D]')

arr.isin([period])
```

**Output:**
```
AttributeError: 'list' object has no attribute 'dtype'
```

**Comparison with IntegerArray (works correctly):**
```python
import pandas as pd

int_arr = pd.array([1, 2, 3], dtype='Int64')
result = int_arr.isin([1])
```

**Output:**
```
<BooleanArray>
[True, False, False]
Length: 3, dtype: boolean
```

## Why This Is A Bug

1. **API Inconsistency**: Other pandas ExtensionArrays (IntegerArray, FloatingArray, BooleanArray) all accept lists in their `isin()` methods
2. **User Expectation**: The `isin()` method signature and documentation suggest it should accept array-like inputs, which includes lists
3. **Unexpected Crash**: Instead of gracefully handling or converting the list, the method crashes with an AttributeError

The error occurs in `/pandas/core/arrays/datetimelike.py:784` where the code attempts to access `values.dtype` without first checking if `values` is a list.

## Fix

The `isin` method in `DatetimeLikeArrayMixin` should convert list inputs to arrays before checking the dtype attribute. Here's the conceptual fix:

```diff
--- a/pandas/core/arrays/datetimelike.py
+++ b/pandas/core/arrays/datetimelike.py
@@ -781,6 +781,10 @@ class DatetimeLikeArrayMixin(ExtensionArray):
         -------
         ndarray[bool]
         """
+        # Convert list to array to ensure we can check dtype
+        if isinstance(values, list):
+            values = np.asarray(values)
+
         if values.dtype.kind in "fiuc":
             return np.zeros(len(self), dtype=bool)
```

Note: This is a conceptual fix. The actual implementation should follow pandas' conventions for type coercion and may need to handle additional edge cases.