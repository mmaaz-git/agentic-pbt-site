# Bug Report: pandas.api.types.pandas_dtype Inconsistent Handling of Series with Object Dtype

**Target**: `pandas.api.types.pandas_dtype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`pandas_dtype()` accepts Series with non-object dtypes (e.g., int64) and returns their dtype, but raises a confusing TypeError when given a Series with object dtype. This violates the consistency principle: the function should handle Series uniformly regardless of their dtype.

## Property-Based Test

```python
import pandas as pd
import pandas.api.types as pat
from hypothesis import given, strategies as st, assume
import numpy as np


@given(st.lists(st.one_of(st.integers(), st.text(), st.none()), min_size=1, max_size=10))
def test_pandas_dtype_should_handle_series_consistently(lst):
    series = pd.Series(lst)

    if series.dtype.kind != 'O':
        assume(False)

    result1 = pat.pandas_dtype(series.dtype)
    result2 = pat.pandas_dtype(series)

    assert result1 == result2, (
        f"pandas_dtype should return the same result for series.dtype and series itself, "
        f"but got {result1} for series.dtype and error for series"
    )
```

**Failing input**: `lst=[None]`

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.types as pat

series_int64 = pd.Series([1, 2, 3])
print(pat.pandas_dtype(series_int64))

series_object = pd.Series([None])
print(pat.pandas_dtype(series_object.dtype))
print(pat.pandas_dtype(series_object))
```

Expected output:
```
int64
object
object
```

Actual output:
```
int64
object
TypeError: dtype '0    None
dtype: object' not understood
```

## Why This Is A Bug

1. **Inconsistent behavior**: The function handles Series with int64 dtype but fails for Series with object dtype
2. **Confusing error message**: The error message prints the entire Series representation instead of a meaningful message
3. **API violation**: Since the function accepts numpy arrays and extracts their dtype, and it accepts Series with some dtypes, it should accept Series with all valid dtypes

The function has special handling for `np.ndarray` but not for `pd.Series`, leading to the Series being passed to `np.dtype()` which converts it to object dtype, then the function rejects it with a confusing error.

## Fix

Add special handling for pandas Series (and other pandas types with dtype attributes) before attempting to convert the input to a dtype:

```diff
def pandas_dtype(dtype) -> DtypeObj:
    """
    Convert input into a pandas only dtype object or a numpy dtype object.

    Parameters
    ----------
    dtype : object to be converted

    Returns
    -------
    np.dtype or a pandas dtype

    Raises
    ------
    TypeError if not a dtype

    Examples
    --------
    >>> pd.api.types.pandas_dtype(int)
    dtype('int64')
    """
    # short-circuit
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
+   elif hasattr(dtype, 'dtype') and hasattr(dtype, '__array__'):
+       # Handle pandas Series, Index, and other array-like objects
+       return dtype.dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype
```

This fix ensures that any object with both a `dtype` attribute and `__array__` method (which includes pandas Series and Index) will have their dtype extracted directly, avoiding the confusing error path.