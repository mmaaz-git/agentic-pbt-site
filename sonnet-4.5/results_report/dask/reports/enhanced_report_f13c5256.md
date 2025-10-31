# Bug Report: dask.dataframe.from_pandas Silently Converts Object Dtype to PyArrow String

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `from_pandas` function silently converts object dtype columns containing strings to string[pyarrow] dtype, violating the round-trip property and contradicting user expectations without any documentation or warning.

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

if __name__ == "__main__":
    test_from_pandas_roundtrip()
```

<details>

<summary>
**Failing input**: `[(0, 0.0, '')]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 27, in <module>
    test_from_pandas_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 6, in test_from_pandas_roundtrip
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_from_pandas_roundtrip
    pd.testing.assert_frame_equal(df_pandas, result)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~^
        lcol,
        ^^^^^
    ...<12 lines>...
        check_flags=False,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 999, in assert_series_equal
    assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    raise_assert_detail(obj, msg, left_attr, right_attr)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Attributes of DataFrame.iloc[:, 2] (column name="c") are different

Attribute "dtype" are different
[left]:  object
[right]: StringDtype(storage=pyarrow, na_value=<NA>)
Falsifying example: test_from_pandas_roundtrip(
    # The test always failed when commented parts were varied together.
    rows=[(0, 0.0, '')],  # or any other generated value
    npartitions=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

# Create a pandas DataFrame with object dtype string columns
df_pandas = pd.DataFrame({'text': ['hello', 'world']})
print(f"Original dtype: {df_pandas['text'].dtype}")
print(f"Original DataFrame:\n{df_pandas}\n")

# Convert to dask DataFrame and back
df_dask = dd.from_pandas(df_pandas, npartitions=1)
result = df_dask.compute()

print(f"After round-trip dtype: {result['text'].dtype}")
print(f"After round-trip DataFrame:\n{result}\n")

# Check if dtypes are preserved
if df_pandas['text'].dtype == result['text'].dtype:
    print("✓ Dtypes match - round-trip successful")
else:
    print(f"✗ Dtype changed from {df_pandas['text'].dtype} to {result['text'].dtype}")

# Now test with minimal input from the Hypothesis test
print("\n--- Testing with minimal input ---")
df_minimal = pd.DataFrame([(0, 0.0, '')], columns=['a', 'b', 'c'])
print(f"Original dtypes:\n{df_minimal.dtypes}\n")

df_dask_minimal = dd.from_pandas(df_minimal, npartitions=1)
result_minimal = df_dask_minimal.compute()

print(f"After round-trip dtypes:\n{result_minimal.dtypes}\n")

# Check if dtypes are preserved for all columns
for col in df_minimal.columns:
    if df_minimal[col].dtype == result_minimal[col].dtype:
        print(f"✓ Column '{col}': dtype preserved ({df_minimal[col].dtype})")
    else:
        print(f"✗ Column '{col}': dtype changed from {df_minimal[col].dtype} to {result_minimal[col].dtype}")
```

<details>

<summary>
Dtype conversion from object to string[pyarrow] demonstrated
</summary>
```
Original dtype: object
Original DataFrame:
    text
0  hello
1  world

After round-trip dtype: string
After round-trip DataFrame:
    text
0  hello
1  world

✗ Dtype changed from object to string

--- Testing with minimal input ---
Original dtypes:
a      int64
b    float64
c     object
dtype: object

After round-trip dtypes:
a              int64
b            float64
c    string[pyarrow]
dtype: object

✓ Column 'a': dtype preserved (int64)
✓ Column 'b': dtype preserved (float64)
✗ Column 'c': dtype changed from object to string
```
</details>

## Why This Is A Bug

This violates the principle of least surprise and the expected round-trip property for several reasons:

1. **Undocumented Behavior**: The `from_pandas` docstring makes no mention that dtypes will be silently converted. Users expect that converting FROM pandas preserves pandas characteristics.

2. **Silent Conversion**: No warning is issued when dtypes are changed. Users cannot detect this change without explicitly checking dtypes after conversion.

3. **Breaks Code Compatibility**: Code that depends on `dtype == 'object'` checks or isinstance checks will fail unexpectedly. This includes:
   - Type checking logic
   - Serialization/deserialization pipelines
   - Code that handles mixed-type object columns differently from pure string columns

4. **Config Option Hidden**: While `dask.config.set({'dataframe.convert-string': False})` can disable this behavior, this option is:
   - Not documented in the `from_pandas` function
   - Not discoverable without reading source code
   - Defaults to `None` which behaves as `True` (confusing)

5. **Violates Round-Trip Expectation**: Users naturally expect that `dd.from_pandas(df).compute()` should return an identical DataFrame, preserving all metadata including dtypes.

## Relevant Context

The root cause is in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_collection.py` at line 4921, where `pyarrow_strings_enabled()` is passed to the `FromPandas` constructor.

The `pyarrow_strings_enabled()` function (in `dask/dataframe/utils.py:782-787`) returns `True` by default when the config value is `None`, enabling automatic conversion:

```python
def pyarrow_strings_enabled() -> bool:
    """Config setting to convert objects to pyarrow strings"""
    convert_string = dask.config.get("dataframe.convert-string")
    if convert_string is None:
        convert_string = True  # Default behavior when not configured
    return convert_string
```

This automatic conversion may provide performance benefits with PyArrow, but it should be opt-in rather than opt-out, or at minimum be clearly documented.

## Proposed Fix

The most straightforward fix is to document this behavior and provide a parameter to control it:

```diff
--- a/dask/dataframe/dask_expr/_collection.py
+++ b/dask/dataframe/dask_expr/_collection.py
@@ -4825,7 +4825,7 @@ def optimize(collection, fuse=True):
     return new_collection(expr.optimize(collection.expr, fuse=fuse))


-def from_pandas(data, npartitions=None, sort=True, chunksize=None):
+def from_pandas(data, npartitions=None, sort=True, chunksize=None, convert_string=None):
     """
     Construct a Dask DataFrame from a Pandas DataFrame

@@ -4836,6 +4836,12 @@ def from_pandas(data, npartitions=None, sort=True, chunksize=None):
     input ordering, make sure the input index is monotonically-increasing. The
     ``sort=False`` option will also avoid reordering, but will not result in
     known divisions.
+
+    .. note::
+       By default, object dtype columns containing strings will be converted to
+       string[pyarrow] dtype for better performance. To preserve original dtypes,
+       set ``convert_string=False`` or configure globally with
+       ``dask.config.set({'dataframe.convert-string': False})``.

     Parameters
     ----------
@@ -4852,6 +4858,10 @@ def from_pandas(data, npartitions=None, sort=True, chunksize=None):
         Sort the input by index first to obtain cleanly divided partitions
         (with known divisions).  If False, the input will not be sorted, and
         all divisions will be set to None. Default is True.
+    convert_string : bool, optional
+        Whether to convert object dtype string columns to pyarrow strings.
+        If None (default), uses the global config setting 'dataframe.convert-string'
+        which defaults to True.

     Returns
     -------
@@ -4912,11 +4922,16 @@ def from_pandas(data, npartitions=None, sort=True, chunksize=None):

     from dask.dataframe.dask_expr.io.io import FromPandas

+    # Use parameter if provided, otherwise fall back to config
+    if convert_string is None:
+        use_pyarrow = pyarrow_strings_enabled()
+    else:
+        use_pyarrow = convert_string
+
     return new_collection(
         FromPandas(
             _BackendData(data.copy()),
             npartitions=npartitions,
             sort=sort,
             chunksize=chunksize,
-            pyarrow_strings_enabled=pyarrow_strings_enabled(),
+            pyarrow_strings_enabled=use_pyarrow,
         )
     )
```