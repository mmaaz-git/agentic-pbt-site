# Bug Report: dask.dataframe.from_pandas Crashes on Unicode Surrogate Characters

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` crashes with a `UnicodeEncodeError` when converting valid pandas DataFrames containing string columns with Unicode surrogate characters (U+D800-U+DFFF), violating its contract to accept any valid pandas DataFrame.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import dask.dataframe as dd


@given(
    data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
        column('c', dtype=str)
    ], index=range_indexes(min_size=1, max_size=50))
)
@settings(max_examples=100, deadline=2000)
def test_from_pandas_roundtrip_dataframe(df):
    ddf = dd.from_pandas(df, npartitions=3)
    result = ddf.compute()
    pd.testing.assert_frame_equal(result, df, check_index_type=False)


if __name__ == "__main__":
    try:
        test_from_pandas_roundtrip_dataframe()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']})`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 23, in <module>
  |     test_from_pandas_roundtrip_dataframe()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 8, in test_from_pandas_roundtrip_dataframe
  |     data_frames([
  |                ^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 16, in test_from_pandas_roundtrip_dataframe
    |     ddf = dd.from_pandas(df, npartitions=3)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_collection.py", line 4915, in from_pandas
    |     return new_collection(
    |         FromPandas(
    |     ...<5 lines>...
    |         )
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_collections.py", line 8, in new_collection
    |     meta = expr._meta
    |            ^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    |     val = self.func(instance)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/io.py", line 445, in _meta
    |     meta = make_meta(to_pyarrow_string(self.frame.head(1)))
    |                      ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/_pyarrow.py", line 69, in _to_string_dtype
    |     df = df.astype(dtypes)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 6639, in astype
    |     res_col = col.astype(dtype=cdt, copy=copy, errors=errors)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 6662, in astype
    |     new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 430, in astype
    |     return self.apply(
    |            ~~~~~~~~~~^
    |         "astype",
    |         ^^^^^^^^^
    |     ...<3 lines>...
    |         using_cow=using_copy_on_write(),
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 363, in apply
    |     applied = getattr(b, f)(**kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/blocks.py", line 784, in astype
    |     new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/astype.py", line 237, in astype_array_safe
    |     new_values = astype_array(values, dtype, copy=copy)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/astype.py", line 182, in astype_array
    |     values = _astype_nansafe(values, dtype, copy=copy)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/astype.py", line 80, in _astype_nansafe
    |     return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/string_arrow.py", line 200, in _from_sequence
    |     return cls(pa.array(result, type=pa.large_string(), from_pandas=True))
    |                ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "pyarrow/array.pxi", line 365, in pyarrow.lib.array
    |   File "pyarrow/array.pxi", line 90, in pyarrow.lib._ndarray_to_array
    |   File "pyarrow/error.pxi", line 89, in pyarrow.lib.check_status
    | UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
    | Falsifying example: test_from_pandas_roundtrip_dataframe(
    |     df=
    |            a    b  c
    |         0  0  0.0  \ud800
    |     ,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 18, in test_from_pandas_roundtrip_dataframe
    |     pd.testing.assert_frame_equal(result, df, check_index_type=False)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    |     assert_series_equal(
    |     ~~~~~~~~~~~~~~~~~~~^
    |         lcol,
    |         ^^^^^
    |     ...<12 lines>...
    |         check_flags=False,
    |         ^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 999, in assert_series_equal
    |     assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    |     ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    |     raise_assert_detail(obj, msg, left_attr, right_attr)
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    |     raise AssertionError(msg)
    | AssertionError: Attributes of DataFrame.iloc[:, 2] (column name="c") are different
    |
    | Attribute "dtype" are different
    | [left]:  StringDtype(storage=pyarrow, na_value=<NA>)
    | [right]: object
    | Falsifying example: test_from_pandas_roundtrip_dataframe(
    |     df=
    |            a    b c
    |         0  0  0.0
    |     ,  # or any other generated value
    | )
    +------------------------------------
Test failed with error: Hypothesis found 2 distinct failures. (2 sub-exceptions)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

# Create a pandas DataFrame with a surrogate character
df = pd.DataFrame({'text': ['\ud800']})
print(f"Pandas DataFrame created successfully with shape: {df.shape}")

# Try to convert to Dask DataFrame
try:
    ddf = dd.from_pandas(df, npartitions=1)
    print("Conversion to Dask DataFrame succeeded")
except Exception as e:
    print(f"Error during conversion: {type(e).__name__}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
UnicodeEncodeError during from_pandas conversion
</summary>
```
Pandas DataFrame created successfully with shape: (1, 1)
Error during conversion: UnicodeEncodeError
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/repo.py", line 10, in <module>
    ddf = dd.from_pandas(df, npartitions=1)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_collection.py", line 4915, in from_pandas
    return new_collection(
        FromPandas(
    ...<5 lines>...
        )
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_collections.py", line 8, in new_collection
    meta = expr._meta
           ^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    val = self.func(instance)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/io.py", line 445, in _meta
    meta = make_meta(to_pyarrow_string(self.frame.head(1)))
                     ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/_pyarrow.py", line 69, in _to_string_dtype
    df = df.astype(dtypes)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 6639, in astype
    res_col = col.astype(dtype=cdt, copy=copy, errors=errors)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 6662, in astype
    new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 430, in astype
    return self.apply(
           ~~~~~~~~~~^
        "astype",
        ^^^^^^^^^
    ...<3 lines>...
        using_cow=using_copy_on_write(),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/blocks.py", line 784, in astype
    new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/astype.py", line 237, in astype_array_safe
    new_values = astype_array(values, dtype, copy=copy)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/astype.py", line 182, in astype_array
    values = _astype_nansafe(values, dtype, copy=copy)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/astype.py", line 80, in _astype_nansafe
    return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/string_arrow.py", line 200, in _from_sequence
    return cls(pa.array(result, type=pa.large_string(), from_pandas=True))
               ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/array.pxi", line 365, in pyarrow.lib.array
  File "pyarrow/array.pxi", line 90, in pyarrow.lib._ndarray_to_array
  File "pyarrow/error.pxi", line 89, in pyarrow.lib.check_status
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Contract violation**: The `from_pandas` function documentation states it accepts "pandas.DataFrame or pandas.Series" without mentioning any restrictions on string content. The function raises TypeError only when "Input must be a pandas DataFrame or Series", implying all valid DataFrames should be accepted.

2. **Valid pandas data**: The input DataFrame is completely valid in pandas. All pandas string operations work correctly with surrogate characters - they can be created, manipulated, and computed without issues.

3. **Silent feature activation**: The PyArrow string conversion happens automatically by default without user awareness. Users don't explicitly request PyArrow strings, yet the conversion crashes on valid data.

4. **Cryptic error message**: The error "UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800'" gives no indication this is related to PyArrow string conversion or that there's a workaround available.

5. **Inconsistent behavior**: The same DataFrame converts successfully when `dataframe.convert-string` is set to False, showing the core functionality works - only the automatic optimization fails.

## Relevant Context

Unicode surrogate characters (U+D800-U+DFFF) are reserved code points that appear in real-world data from:
- Corrupted UTF-8 data from web scraping
- Binary data incorrectly interpreted as text
- Legacy systems with non-standard encodings
- Database exports with encoding issues

The bug occurs in `/dask/dataframe/_pyarrow.py:69` where `df.astype(dtypes)` attempts to convert object dtype strings to PyArrow string dtype. PyArrow enforces strict UTF-8 validation and rejects surrogate characters as invalid.

**Workaround**: Users can disable PyArrow string conversion:
```python
import dask
dask.config.set({'dataframe.convert-string': False})
```

This allows the conversion to succeed, but users must discover this undocumented workaround through trial and error.

Related files:
- Bug location: `/dask/dataframe/_pyarrow.py:69` in `_to_string_dtype` function
- Called from: `/dask/dataframe/dask_expr/io/io.py:445` in FromPandas._meta property
- Config check: `/dask/dataframe/utils.py` in `pyarrow_strings_enabled()` function

## Proposed Fix

Add error handling to gracefully handle PyArrow conversion failures and provide helpful guidance to users:

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -66,7 +66,15 @@ def _to_string_dtype(df, dtype_check, index_check, string_dtype):
             col: string_dtype for col, dtype in df.dtypes.items() if dtype_check(dtype)
         }
         if dtypes:
-            df = df.astype(dtypes)
+            try:
+                df = df.astype(dtypes)
+            except (UnicodeEncodeError, Exception) as e:
+                if "surrogates not allowed" in str(e):
+                    raise ValueError(
+                        f"Cannot convert to PyArrow string dtype due to invalid UTF-8 characters. "
+                        f"Your data contains Unicode surrogate characters that PyArrow cannot encode. "
+                        f"To fix this, disable PyArrow string conversion: "
+                        f"dask.config.set({{'dataframe.convert-string': False}})"
+                    ) from e
+                raise
     elif dtype_check(df.dtype):
         dtypes = string_dtype
         df = df.copy().astype(dtypes)
```