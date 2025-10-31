# Bug Report: dask.dataframe.from_pandas Silently Converts String Dtypes

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` followed by `.compute()` does not preserve the original dtypes of string columns. Object-dtype string columns are silently converted to PyArrow-backed StringDtype, violating the round-trip property.

## Property-Based Test

```python
import dask.dataframe as dd
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

@given(
    data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
        column('c', dtype=str),
    ], index=range_indexes(min_size=1, max_size=100))
)
def test_from_pandas_compute_roundtrip(pdf):
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()
    pd.testing.assert_frame_equal(result, pdf)
```

**Failing input**: Any pandas DataFrame with string columns (dtype=object)

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

pdf = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['hello']})

print("Input dtype:", pdf['c'].dtype)

ddf = dd.from_pandas(pdf, npartitions=2)
result = ddf.compute()

print("Output dtype:", result['c'].dtype)

assert pdf['c'].dtype == result['c'].dtype
```

Expected: Both dtypes should be `object`
Actual:
- Input dtype: `object`
- Output dtype: `string[pyarrow]`
- Assertion fails

## Why This Is A Bug

The `from_pandas` function documentation does not mention that it will modify dtypes. A reasonable expectation for `from_pandas(pdf).compute()` is that it returns a DataFrame equivalent to the input. This round-trip property is fundamental for data processing workflows.

Users may depend on specific dtypes for:
1. Compatibility with other libraries that expect object dtypes
2. Serialization/deserialization workflows
3. Handling edge cases (like surrogate characters) that PyArrow strings don't support

The silent conversion can cause unexpected behavior, especially since PyArrow strings have different semantics than object-dtype strings (e.g., NaN vs None for missing values, Unicode handling).

## Fix

The conversion to PyArrow strings happens in `_to_string_dtype` in `dask/dataframe/_pyarrow.py`. Dask should either:

1. Make the conversion opt-in via a parameter (e.g., `convert_string_to_pyarrow=False` by default)
2. Document this behavior clearly in the `from_pandas` docstring
3. Preserve the original dtypes by default

Option 3 (preserve dtypes) would best maintain the round-trip property:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -442,7 +442,7 @@ class FromPandas(IO):

     @functools.cached_property
     def _meta(self):
-        meta = make_meta(to_pyarrow_string(self.frame.head(1)))
+        meta = make_meta(self.frame.head(1))
         return meta
```

This would prevent the automatic conversion to PyArrow strings and preserve the original object dtype.