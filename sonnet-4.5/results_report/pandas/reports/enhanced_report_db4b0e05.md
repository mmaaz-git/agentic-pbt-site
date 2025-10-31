# Bug Report: pandas.core.interchange Object Dtype Integers Cause Delayed NotImplementedError

**Target**: `pandas.core.interchange`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The DataFrame interchange protocol fails to validate unsupported object dtype integers during `__dataframe__()` creation, instead failing later during `from_dataframe()` with a NotImplementedError, violating the fail-fast principle.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.interchange import from_dataframe

@given(
    data=st.lists(st.integers(), min_size=0, max_size=100),
    col_name=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
@settings(max_examples=1000)
def test_round_trip_integer_column(data, col_name):
    df = pd.DataFrame({col_name: data})

    interchange_df = df.__dataframe__()
    df_roundtrip = from_dataframe(interchange_df)

    assert df.equals(df_roundtrip), f"Round-trip failed: {df} != {df_roundtrip}"

# Run the test
test_round_trip_integer_column()
```

<details>

<summary>
**Failing input**: `data=[-9_223_372_036_854_775_809], col_name='A'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 19, in <module>
    test_round_trip_integer_column()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 6, in test_round_trip_integer_column
    data=st.lists(st.integers(), min_size=0, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 14, in test_round_trip_integer_column
    df_roundtrip = from_dataframe(interchange_df)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 100, in from_dataframe
    return _from_dataframe(
        df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 123, in _from_dataframe
    pandas_df = protocol_df_chunk_to_pandas(chunk)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 164, in protocol_df_chunk_to_pandas
    dtype = col.dtype[0]
            ^^^^^^^^^
  File "pandas/_libs/properties.pyx", line 36, in pandas._libs.properties.CachedProperty.__get__
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 144, in dtype
    raise NotImplementedError("Non-string object dtypes are not supported yet")
NotImplementedError: Non-string object dtypes are not supported yet
Falsifying example: test_round_trip_integer_column(
    data=[-9_223_372_036_854_775_809],
    col_name='A',  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:329
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py:6312
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:5461
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:5462
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py:137
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/series.py:1044
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

# Create a DataFrame with an integer that's outside int64 range
# int64 min is -9,223,372,036,854,775,808
# We use -9,223,372,036,854,775,809 (one less than min)
df = pd.DataFrame({'a': [-9_223_372_036_854_775_809]})

print(f"DataFrame dtype: {df['a'].dtype}")
print(f"Value is below int64 min: {-9_223_372_036_854_775_809 < np.iinfo(np.int64).min}")

# This succeeds
interchange_df = df.__dataframe__()
print("Created interchange object successfully")

# This fails with NotImplementedError
try:
    df_roundtrip = from_dataframe(interchange_df)
    print("Successfully converted from interchange")
except NotImplementedError as e:
    print(f"Failed with error: {e}")
```

<details>

<summary>
NotImplementedError during from_dataframe() call
</summary>
```
DataFrame dtype: object
Value is below int64 min: True
Created interchange object successfully
Failed with error: Non-string object dtypes are not supported yet
```
</details>

## Why This Is A Bug

This violates the fail-fast principle in API design. When pandas encounters integers that exceed the int64 range (e.g., -9,223,372,036,854,775,809), it correctly stores them using object dtype. However, the interchange protocol implementation has an inconsistent validation strategy:

1. **`__dataframe__()`** succeeds without validation, creating a `PandasDataFrameXchg` object
2. **`from_dataframe()`** fails when it tries to access the column's dtype property
3. The error occurs in `/pandas/core/interchange/column.py:144` where it checks if object dtype columns contain strings, and raises `NotImplementedError` for non-string object dtypes

This creates a partially-constructed interchange object that appears valid but cannot be used, leading to confusing delayed errors. Users have no way to know in advance that their data is unsupported until they attempt to use the interchange object.

## Relevant Context

The bug occurs specifically at line 136-144 in `/pandas/core/interchange/column.py`:

```python
elif is_string_dtype(dtype):
    if infer_dtype(self._col) in ("string", "empty"):
        return (
            DtypeKind.STRING,
            8,
            dtype_to_arrow_c_fmt(dtype),
            Endianness.NATIVE,
        )
    raise NotImplementedError("Non-string object dtypes are not supported yet")
```

When a column has object dtype, the code checks if it contains strings. If `infer_dtype()` returns "integer" (as it does for large integers), it raises the NotImplementedError. This check happens lazily when the dtype property is accessed, not during interchange object creation.

Python's arbitrary-precision integers are a core language feature, and pandas correctly handles them with object dtype. Financial and scientific applications commonly use such large integers. The interchange protocol documentation doesn't mention this limitation.

## Proposed Fix

Add early validation in `DataFrame.__dataframe__()` to reject unsupported object dtypes immediately:

```diff
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -981,6 +981,16 @@ class DataFrame(NDFrame, OpsMixin):
         """

         from pandas.core.interchange.dataframe import PandasDataFrameXchg
+        from pandas._libs.lib import infer_dtype
+        from pandas.api.types import is_string_dtype
+
+        for col in self.columns:
+            dtype = self[col].dtype
+            if is_string_dtype(dtype) and dtype == object:
+                inferred = infer_dtype(self[col])
+                if inferred not in ("string", "empty"):
+                    raise ValueError(
+                        f"Column '{col}' has unsupported object dtype (inferred: {inferred}). "
+                        "The interchange protocol only supports object columns containing strings."
+                    )

         return PandasDataFrameXchg(self, allow_copy=allow_copy)
```