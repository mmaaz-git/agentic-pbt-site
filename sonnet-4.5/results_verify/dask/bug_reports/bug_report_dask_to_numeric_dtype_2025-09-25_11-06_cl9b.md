# Bug Report: dask.dataframe.to_numeric Returns Wrong Dtype

**Target**: `dask.dataframe.to_numeric`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.dataframe.to_numeric` returns nullable dtypes (`Int64`, `Float64`) instead of numpy dtypes (`int64`, `float64`) as documented, violating its API contract and differing from pandas behavior.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd
import dask.dataframe as dd

@given(st.lists(st.integers(min_value=-1000, max_value=1000).map(str), min_size=1, max_size=100))
@settings(max_examples=100)
def test_to_numeric_dtype_matches_pandas(values):
    s = pd.Series(values)
    ds = dd.from_pandas(s, npartitions=2)

    result_pandas = pd.to_numeric(s)
    result_dask = dd.to_numeric(ds).compute()

    assert result_dask.dtype == result_pandas.dtype, \
        f"Expected dtype {result_pandas.dtype}, got {result_dask.dtype}"
```

**Failing input**: `['0']`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

s = pd.Series(['0', '1', '2'])
ds = dd.from_pandas(s, npartitions=2)

result_pandas = pd.to_numeric(s)
result_dask = dd.to_numeric(ds).compute()

print(f'Pandas dtype: {result_pandas.dtype}')
print(f'Dask dtype: {result_dask.dtype}')

assert result_dask.dtype == result_pandas.dtype
```

Output:
```
Pandas dtype: int64
Dask dtype: Int64
AssertionError
```

## Why This Is A Bug

The `dask.dataframe.to_numeric` docstring explicitly states:

> The default return dtype is `float64` or `int64` depending on the data supplied.

However, dask returns `Int64` (pandas nullable integer dtype) instead of `int64` (numpy dtype), and `Float64` instead of `float64`. This violates the documented API contract and creates incompatibility with pandas. The docstring also says "This docstring was copied from pandas.to_numeric" and "Some inconsistencies with the Dask version may exist", but the dtype difference is significant enough that it should either be:

1. Fixed to match pandas behavior, or
2. Explicitly documented as a known difference

This affects users who expect pandas-compatible behavior and may rely on the specific dtype (e.g., for interoperability with numpy operations that don't support nullable dtypes).

## Fix

The issue appears to be in the `ToNumeric` expression class. The fix would involve ensuring that the output dtype matches pandas' behavior by using numpy dtypes instead of nullable dtypes by default. This would likely require modifications to how the result dtype is inferred in the `ToNumeric` class implementation.

A high-level fix would involve:
1. Identifying where `ToNumeric` sets the output dtype
2. Changing it to use numpy dtypes (`int64`, `float64`) instead of nullable dtypes (`Int64`, `Float64`)
3. Ensuring this doesn't break the `errors='coerce'` case, which should still use `float64` (not `Float64`) for NaN-containing results
