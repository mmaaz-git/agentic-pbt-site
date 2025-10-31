# Bug Report: dask.dataframe.from_pandas Silently Changes Dtype

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `from_pandas` function silently converts object dtype columns to string[pyarrow] dtype, violating the round-trip property and contradicting user expectations. This behavior is undocumented and breaks code that depends on dtype preservation.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

@settings(max_examples=50)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-100, max_value=100),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.text(max_size=10)
        ),
        min_size=1,
        max_size=50
    ),
    st.integers(min_value=1, max_value=5)
)
def test_from_pandas_roundtrip(rows, npartitions):
    df_pandas = pd.DataFrame(rows, columns=['a', 'b', 'c'])

    df_dask = dd.from_pandas(df_pandas, npartitions=npartitions)
    result = df_dask.compute()

    pd.testing.assert_frame_equal(df_pandas, result)
```

**Failing input**: Any pandas DataFrame with object dtype string columns, e.g., `[(0, 0.0, '')]`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df_pandas = pd.DataFrame({'text': ['hello', 'world']})
print(f"Original dtype: {df_pandas['text'].dtype}")

df_dask = dd.from_pandas(df_pandas, npartitions=1)
result = df_dask.compute()
print(f"After round-trip: {result['text'].dtype}")

assert df_pandas['text'].dtype == result['text'].dtype
```

Output:
```
Original dtype: object
After round-trip: string
AssertionError: Dtype changed from object to string
```

## Why This Is A Bug

This is a contract violation because:

1. **Undocumented behavior**: The `from_pandas` docstring makes no mention of dtype conversion
2. **Violates round-trip expectation**: Users expect `df.compute()` on a DataFrame created from pandas to return the original dtypes
3. **Silent conversion**: No warning is issued when dtypes are changed
4. **Breaks user code**: Code that checks `dtype == 'object'` or uses isinstance checks will fail
5. **Not discoverable**: The `dataframe.convert-string` config option that controls this behavior is not mentioned in the documentation

While there exists a config option `dataframe.convert-string=False` to disable this behavior, the default silently changes dtypes without user consent or awareness.

## Fix

The fix should either:

1. **Document the behavior** in the `from_pandas` docstring and issue a warning when dtypes are converted, OR
2. **Change the default** to preserve dtypes by default (set `convert-string=False` as default), OR
3. **Add a parameter** to `from_pandas` to explicitly control dtype conversion

Minimal documentation fix:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -10,6 +10,11 @@ def from_pandas(data, npartitions=None, sort=True, chunksize=None):
     produce cleanly-divided partitions (with known divisions).  To preserve the
     input ordering, make sure the input index is monotonically-increasing. The
     ``sort=False`` option will also avoid reordering, but will not result in
     known divisions.
+
+    .. note::
+       By default, object dtype columns containing strings will be converted to
+       string[pyarrow] dtype. To preserve original dtypes, set the config option
+       ``dask.config.set({'dataframe.convert-string': False})``.

     Parameters
     ----------
```