# Bug Report: dask.dataframe.from_pandas Silently Converts String Dtypes

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` silently converts object-dtype string columns to PyArrow-backed string dtype, violating the round-trip property and causing crashes with certain Unicode characters.

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

if __name__ == "__main__":
    # Run the test
    test_from_pandas_compute_roundtrip()
```

<details>

<summary>
**Failing input**: `pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['']})` (or any DataFrame with string columns)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 20, in <module>
  |     test_from_pandas_compute_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 7, in test_from_pandas_compute_roundtrip
  |     data_frames([
  |                ^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 14, in test_from_pandas_compute_roundtrip
    |     ddf = dd.from_pandas(pdf, npartitions=2)
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 16, in test_from_pandas_compute_roundtrip
    |     pd.testing.assert_frame_equal(result, pdf)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
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
    | Falsifying example: test_from_pandas_compute_roundtrip(
    |     pdf=
    |            a    b c
    |         0  0  0.0
    |     ,  # or any other generated value
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

# Create a simple pandas DataFrame with string columns
pdf = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['hello']})

print("Original pandas DataFrame:")
print(pdf)
print("\nOriginal dtypes:")
for col in pdf.columns:
    print(f"  {col}: {pdf[col].dtype}")

# Convert to Dask DataFrame and back
ddf = dd.from_pandas(pdf, npartitions=2)
result = ddf.compute()

print("\nResult after dd.from_pandas().compute():")
print(result)
print("\nResult dtypes:")
for col in result.columns:
    print(f"  {col}: {result[col].dtype}")

# Check if dtypes match
print("\nDtype comparison:")
for col in pdf.columns:
    original_dtype = pdf[col].dtype
    result_dtype = result[col].dtype
    matches = original_dtype == result_dtype
    print(f"  Column '{col}': {original_dtype} -> {result_dtype} (Match: {matches})")

# Assertion that fails
print("\nAssertion check for string column 'c':")
try:
    assert pdf['c'].dtype == result['c'].dtype, f"Dtype mismatch: {pdf['c'].dtype} != {result['c'].dtype}"
    print("  PASSED: Dtypes match")
except AssertionError as e:
    print(f"  FAILED: {e}")
```

<details>

<summary>
AssertionError: Dtype mismatch for string column
</summary>
```
Original pandas DataFrame:
   a    b      c
0  0  0.0  hello

Original dtypes:
  a: int64
  b: float64
  c: object

Result after dd.from_pandas().compute():
   a    b      c
0  0  0.0  hello

Result dtypes:
  a: int64
  b: float64
  c: string

Dtype comparison:
  Column 'a': int64 -> int64 (Match: True)
  Column 'b': float64 -> float64 (Match: True)
  Column 'c': object -> string (Match: False)

Assertion check for string column 'c':
  FAILED: Dtype mismatch: object != string
```
</details>

## Why This Is A Bug

This violates the fundamental round-trip property that `from_pandas(df).compute()` should return a DataFrame equivalent to the original. The issues are:

1. **Undocumented behavior**: The `from_pandas` docstring makes no mention of dtype conversion. Users reasonably expect the function to preserve dtypes.

2. **Silent conversion**: The conversion happens automatically without any warning or parameter to control it. The behavior is controlled by an undocumented config setting `dataframe.convert-string` that defaults to `True`.

3. **Breaking changes**: The conversion causes real problems:
   - PyArrow strings crash on surrogate Unicode characters (e.g., '\ud800'), as shown in the hypothesis test
   - Different handling of missing values (NaN vs pd.NA)
   - Breaks compatibility with code expecting object dtypes
   - No way to know this will happen without reading source code

4. **Inconsistent API**: The function is named `from_pandas` implying it creates a Dask version of the pandas DataFrame, not a modified one with different dtypes.

## Relevant Context

The conversion happens in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/io.py:443-445`:

```python
@functools.cached_property
def _meta(self):
    if self.pyarrow_strings_enabled:
        meta = make_meta(to_pyarrow_string(self.frame.head(1)))
    else:
        meta = self.frame.head(0)
```

The `pyarrow_strings_enabled()` function in `dask/dataframe/utils.py:782-787` checks the config:

```python
def pyarrow_strings_enabled() -> bool:
    """Config setting to convert objects to pyarrow strings"""
    convert_string = dask.config.get("dataframe.convert-string")
    if convert_string is None:
        convert_string = True
    return convert_string
```

The workaround is to set `dask.config.set({"dataframe.convert-string": False})` before using `from_pandas`.

## Proposed Fix

The conversion should be opt-in rather than opt-out to preserve backward compatibility and the principle of least surprise:

```diff
--- a/dask/dataframe/utils.py
+++ b/dask/dataframe/utils.py
@@ -783,7 +783,7 @@ def pyarrow_strings_enabled() -> bool:
     """Config setting to convert objects to pyarrow strings"""
     convert_string = dask.config.get("dataframe.convert-string")
     if convert_string is None:
-        convert_string = True
+        convert_string = False
     return convert_string
```

Alternatively, add an explicit parameter to `from_pandas`:

```diff
--- a/dask/dataframe/dask_expr/_collection.py
+++ b/dask/dataframe/dask_expr/_collection.py
@@ -4890,6 +4890,7 @@ def from_pandas(
     npartitions: int | None = None,
     chunksize: int | None = None,
     sort: bool = True,
+    convert_string_to_pyarrow: bool | None = None,
 ) -> DataFrame | Series:
     """Construct a Dask DataFrame from a Pandas DataFrame

@@ -4908,6 +4909,9 @@ def from_pandas(
     sort : bool
         Sort the index of the DataFrame. This defaults to True if the index is
         not already sorted.
+    convert_string_to_pyarrow : bool, optional
+        Whether to convert object-dtype string columns to PyArrow strings.
+        If None, uses the 'dataframe.convert-string' config setting.

     Returns
     -------
```