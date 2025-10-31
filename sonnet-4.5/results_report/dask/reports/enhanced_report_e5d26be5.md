# Bug Report: dask.dataframe.dask_expr Integer Overflow to String Conversion

**Target**: `dask.dataframe.dask_expr.from_pandas`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `from_pandas` function incorrectly converts pandas Series containing large integers (that overflow int64) from object dtype to PyArrow string dtype, causing data corruption where integer values become strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import dask.dataframe.dask_expr as dex

@settings(max_examples=100)
@given(
    st.lists(st.integers(), min_size=1, max_size=50),
    st.integers(min_value=1, max_value=5)
)
def test_from_pandas_round_trip_series(data, npartitions):
    s = pd.Series(data)
    dask_s = dex.from_pandas(s, npartitions=npartitions, sort=False)
    result = dask_s.compute()
    pd.testing.assert_series_equal(result, s, check_index_type=False)

# Run the test
test_from_pandas_round_trip_series()
```

<details>

<summary>
**Failing input**: `data=[-9223372036854775809], npartitions=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 17, in <module>
    test_from_pandas_round_trip_series()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 6, in test_from_pandas_round_trip_series
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 14, in test_from_pandas_round_trip_series
    pd.testing.assert_series_equal(result, s, check_index_type=False)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 999, in assert_series_equal
    assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    raise_assert_detail(obj, msg, left_attr, right_attr)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Attributes of Series are different

Attribute "dtype" are different
[left]:  StringDtype(storage=pyarrow, na_value=<NA>)
[right]: object
Falsifying example: test_from_pandas_round_trip_series(
    data=[-9_223_372_036_854_775_809],
    npartitions=1,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/_pyarrow.py:26
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/_pyarrow.py:71
        /home/npc/miniconda/lib/python3.13/site-packages/dask/tokenize.py:446
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/__init__.py:153
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:113
        (and 33 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe.dask_expr as dex

# Create a Series with an integer that overflows int64
s = pd.Series([-9223372036854775809])
print(f"Original dtype: {s.dtype}")
print(f"Original value: {s.values[0]} (type: {type(s.values[0]).__name__})")
print(f"Original Series:\n{s}")
print()

# Convert to Dask and back
dask_s = dex.from_pandas(s, npartitions=1, sort=False)
result = dask_s.compute()

print(f"Result dtype: {result.dtype}")
print(f"Result value: {result.values[0]} (type: {type(result.values[0]).__name__})")
print(f"Result Series:\n{result}")
print()

# Check if they are equal
print(f"Are dtypes equal? {s.dtype == result.dtype}")
print(f"Are values equal? {s.values[0] == result.values[0]}")

# This should be True but will be False due to the bug
assert s.dtype == result.dtype, f"Expected dtype {s.dtype}, got {result.dtype}"
```

<details>

<summary>
AssertionError: Expected dtype object, got string
</summary>
```
Original dtype: object
Original value: -9223372036854775809 (type: int)
Original Series:
0    -9223372036854775809
dtype: object

Result dtype: string
Result value: -9223372036854775809 (type: str)
Result Series:
0    -9223372036854775809
dtype: string

Are dtypes equal? False
Are values equal? False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/repo.py", line 25, in <module>
    assert s.dtype == result.dtype, f"Expected dtype {s.dtype}, got {result.dtype}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected dtype object, got string
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Data Type Corruption**: The function silently changes the fundamental data type from integer to string. The value `-9223372036854775809` is an integer that cannot fit in int64, so pandas stores it as a Python int in an object dtype Series. After round-tripping through `from_pandas`, it becomes a string `"-9223372036854775809"`.

2. **Violates Round-Trip Property**: The `from_pandas` function documentation states it "constructs a dask.dataframe from those parts" and shows examples where the original data is preserved. Users reasonably expect that `from_pandas(s).compute()` should return data equivalent to the original Series `s`.

3. **Silent Failure**: No warning or error is raised when this data corruption occurs. Users may not discover their integer data has been converted to strings until downstream operations fail with type errors.

4. **Inconsistent Behavior**: The conversion only happens when PyArrow strings are enabled (the default) and only affects object dtypes containing certain data types, creating unpredictable behavior.

5. **Breaks Mathematical Operations**: After conversion, mathematical operations that worked on the original integer data will fail or produce incorrect results on the string data.

## Relevant Context

The root cause is in the interaction between two components:

1. **`/dask/dataframe/dask_expr/io/io.py`**: The `FromPandas` class (lines 407-551) checks if `pyarrow_strings_enabled` is True (line 444, 535) and if so, calls `to_pyarrow_string` on the data.

2. **`/dask/dataframe/_pyarrow.py`**: The `is_object_string_dtype` function (lines 20-27) incorrectly identifies object dtypes as string dtypes using `pd.api.types.is_string_dtype(dtype)` which returns True for object dtypes even when they contain non-string data like large integers.

The issue occurs because `pd.api.types.is_string_dtype` is overly permissive - it returns True for any object dtype since object dtypes *could* contain strings, even when they actually contain integers.

Related documentation:
- PyArrow string conversion is documented at: https://docs.dask.org/en/stable/dataframe-api.html#text-data
- The `from_pandas` function: https://docs.dask.org/en/stable/generated/dask.dataframe.from_pandas.html

Similar issues have been reported: dask/dask#10546 shows related problems with PyArrow string conversion.

## Proposed Fix

The fix should modify `is_object_string_dtype` in `/dask/dataframe/_pyarrow.py` to properly detect whether an object dtype actually contains string data rather than assuming all object dtypes are strings:

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -19,10 +19,16 @@ def is_pyarrow_string_dtype(dtype):

 def is_object_string_dtype(dtype):
     """Determine if input is a non-pyarrow string dtype"""
+    # Don't automatically convert object dtypes - they may contain non-string data
+    # like large integers that overflow int64, mixed types, or custom objects
+    if pd.api.types.is_object_dtype(dtype):
+        return False
+
     # in pandas < 2.0, is_string_dtype(DecimalDtype()) returns True
     return (
         pd.api.types.is_string_dtype(dtype)
         and not is_pyarrow_string_dtype(dtype)
         and not pd.api.types.is_dtype_equal(dtype, "decimal")
     )
```

This conservative fix prevents automatic conversion of object dtypes to PyArrow strings, avoiding data corruption. A more sophisticated solution could sample the actual data to determine if it truly contains strings, but that would require passing data to this function and could have performance implications.