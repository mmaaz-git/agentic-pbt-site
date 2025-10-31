# Bug Report: dask.dataframe.from_pandas Unicode Surrogate Crash

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` crashes with a UnicodeEncodeError when the input pandas DataFrame contains Unicode surrogate characters in string columns, even though pandas itself supports storing such characters.

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
    """Test that from_pandas can handle any valid pandas DataFrame"""
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()
    pd.testing.assert_frame_equal(result, pdf)

if __name__ == "__main__":
    # Run the property test
    test_from_pandas_compute_roundtrip()
```

<details>

<summary>
**Failing input**: DataFrame with column 'c' containing '\ud800' (Unicode surrogate)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/39
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_from_pandas_compute_roundtrip FAILED

=================================== FAILURES ===================================
______________________ test_from_pandas_compute_roundtrip ______________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 7, in test_from_pandas_compute_roundtrip
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 15, in test_from_pandas_compute_roundtrip
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
    | Falsifying example: test_from_pandas_compute_roundtrip(
    |     pdf=
    |            a    b  c
    |         0  0  0.0  \ud800
    |     ,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 17, in test_from_pandas_compute_roundtrip
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
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    |     raise AssertionError(msg)
    | AssertionError: Attributes of DataFrame.iloc[:, 2] (column name="c") are different
    |
    | Attribute "dtype" are different
    | [left]:  StringDtype(storage=pyarrow, na_value=<NA>)
    | [right]: object
    | Falsifying example: test_from_pandas_compute_roundtrip(
    |     pdf=
    |            a    b c
    |         0  0  0.0
    |     ,  # or any other generated value
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_from_pandas_compute_roundtrip - ExceptionGroup: Hypothes...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!
============================== 1 failed in 13.76s ==============================
```
</details>

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

# Create a pandas DataFrame with a Unicode surrogate character
# Surrogate characters are half of a UTF-16 surrogate pair (U+D800 to U+DFFF)
# They are not valid standalone Unicode characters and cannot be encoded to UTF-8
pdf = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']})

print("Pandas DataFrame created successfully:")
print(f"DataFrame shape: {pdf.shape}")
print(f"Columns: {pdf.columns.tolist()}")
print(f"String column value (repr): {repr(pdf['c'].iloc[0])}")

# Try to create a Dask DataFrame from the pandas DataFrame
print("\nTrying to create Dask DataFrame...")
try:
    ddf = dd.from_pandas(pdf, npartitions=2)
    print("Dask DataFrame created successfully")
    result = ddf.compute()
    print("Computed result:")
    print(result)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800'
</summary>
```
Pandas DataFrame created successfully:
DataFrame shape: (1, 3)
Columns: ['a', 'b', 'c']
String column value (repr): '\ud800'

Trying to create Dask DataFrame...
ERROR: UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/repo.py", line 17, in <module>
    ddf = dd.from_pandas(pdf, npartitions=2)
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

This violates the expected behavior of `from_pandas` because:

1. **Pandas DataFrames support Unicode surrogates**: Pandas can store and manipulate strings containing surrogate characters (U+D800-U+DFFF) without issues, as they use Python's native string type which allows unpaired surrogates.

2. **from_pandas should accept all valid pandas DataFrames**: The function's contract is to convert any pandas DataFrame into a Dask DataFrame. There is no documented restriction about surrogate characters.

3. **The error is internal and not user-facing**: The crash occurs during internal dtype conversion when Dask automatically tries to convert object dtype strings to PyArrow strings. This automatic conversion is an optimization, not a required operation.

4. **PyArrow limitation is not Dask limitation**: While PyArrow strings require valid UTF-8 (which excludes surrogates), Dask should gracefully handle this limitation by falling back to object dtype rather than crashing.

5. **Workaround exists but is not obvious**: Users can disable pyarrow strings globally with `dask.config.set({"dataframe.convert-string": False})`, but this is not discoverable from the error message.

## Relevant Context

- The error occurs in `/dask/dataframe/dask_expr/io/io.py:445` when creating metadata for the new Dask DataFrame
- The `to_pyarrow_string()` function in `/dask/dataframe/_pyarrow.py:69` attempts to convert object dtype to PyArrow string dtype
- PyArrow strings require valid UTF-8 encoding, which prohibits surrogate characters
- The conversion happens automatically when `pyarrow_strings_enabled()` returns True (the default)
- Related code: https://github.com/dask/dask/blob/main/dask/dataframe/_pyarrow.py
- Documentation on string dtypes: https://docs.dask.org/en/latest/dataframe-design.html#text-data

## Proposed Fix

The issue should be fixed by catching the UnicodeEncodeError in `_to_string_dtype` and falling back to the original dtype:

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -66,7 +66,11 @@ def _to_string_dtype(df, dtype_check, index_check, string_dtype):
             col: string_dtype for col, dtype in df.dtypes.items() if dtype_check(dtype)
         }
         if dtypes:
-            df = df.astype(dtypes)
+            try:
+                df = df.astype(dtypes)
+            except (UnicodeEncodeError, UnicodeDecodeError):
+                # Fallback to object dtype if conversion fails (e.g., surrogate characters)
+                pass
     elif dtype_check(df.dtype):
         dtypes = string_dtype
-        df = df.copy().astype(dtypes)
+        try:
+            df = df.copy().astype(dtypes)
+        except (UnicodeEncodeError, UnicodeDecodeError):
+            # Keep original dtype if conversion fails
+            pass

     # Convert DataFrame/Series index too
     if (is_dataframe_like(df) or is_series_like(df)) and index_check(df.index):
@@ -77,14 +81,18 @@ def _to_string_dtype(df, dtype_check, index_check, string_dtype):
             levels = {
                 i: level.astype(string_dtype)
                 for i, level in enumerate(df.index.levels)
                 if dtype_check(level.dtype)
             }
             # set verify_integrity=False to preserve index codes
-            df.index = df.index.set_levels(
-                levels.values(), level=levels.keys(), verify_integrity=False
-            )
+            try:
+                df.index = df.index.set_levels(
+                    levels.values(), level=levels.keys(), verify_integrity=False
+                )
+            except (UnicodeEncodeError, UnicodeDecodeError):
+                pass
         else:
-            df.index = df.index.astype(string_dtype)
+            try:
+                df.index = df.index.astype(string_dtype)
+            except (UnicodeEncodeError, UnicodeDecodeError):
+                pass
     return df
```