# Bug Report: pandas.api.interchange.from_dataframe UnicodeEncodeError with Surrogate Characters

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas DataFrame interchange protocol crashes with a UnicodeEncodeError when attempting to convert DataFrames containing string columns with Unicode surrogate characters (U+D800 to U+DFFF) which are valid Python strings but cannot be encoded to UTF-8.

## Property-Based Test

```python
#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from hypothesis.extra.pandas import data_frames, column
import pandas as pd

# Create a strategy that can generate surrogate characters
surrogate_text = st.text(alphabet=st.characters(min_codepoint=0xD800, max_codepoint=0xDFFF), min_size=1, max_size=1)

@given(data_frames(columns=[
    column('int_col', dtype=int),
    column('float_col', dtype=float),
    column('str_col', dtype=str)
]))
@example(pd.DataFrame({
    'int_col': [0],
    'float_col': [0.0],
    'str_col': ['\ud800']
}))
@settings(max_examples=100)
def test_round_trip_mixed_types(df):
    """Round-trip should preserve mixed column types."""
    interchange_obj = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(interchange_obj)

    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)

if __name__ == "__main__":
    # Run the test
    test_round_trip_mixed_types()
```

<details>

<summary>
**Failing input**: `pd.DataFrame({'int_col': [0], 'float_col': [0.0], 'str_col': ['\ud800']})`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 34, in <module>
    test_round_trip_mixed_types()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 14, in test_round_trip_mixed_types
    column('int_col', dtype=int),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 27, in test_round_trip_mixed_types
    result = pd.api.interchange.from_dataframe(interchange_obj)
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 100, in from_dataframe
    return _from_dataframe(
        df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy
    )
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 123, in _from_dataframe
    pandas_df = protocol_df_chunk_to_pandas(chunk)
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 175, in protocol_df_chunk_to_pandas
    columns[name], buf = string_column_to_ndarray(col)
                         ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 292, in string_column_to_ndarray
    buffers = col.get_buffers()
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 287, in get_buffers
    "data": self._get_data_buffer(),
            ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 351, in _get_data_buffer
    b.extend(obj.encode(encoding="utf-8"))
             ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
Falsifying explicit example: test_round_trip_mixed_types(
    df=
           int_col  float_col str_col
        0        0        0.0       \ud800
    ,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd

# Create a DataFrame with surrogate character '\ud800'
df = pd.DataFrame({
    'int_col': [0],
    'float_col': [0.0],
    'str_col': ['\ud800']
})

print("DataFrame created successfully")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"DataFrame dtypes:")
print(df.dtypes)
print(f"String column contains surrogate character: U+D800")

# Get the interchange object
interchange_obj = df.__dataframe__()
print(f"\nInterchange object created successfully")

# Attempt to convert back using from_dataframe
print("\nAttempting to convert back using pd.api.interchange.from_dataframe()...")
try:
    result = pd.api.interchange.from_dataframe(interchange_obj)
    print(f"Result DataFrame created successfully")
    print(f"Shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
except Exception as e:
    import traceback
    print(f"\nError occurred: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
UnicodeEncodeError when encoding surrogate character to UTF-8
</summary>
```
DataFrame created successfully
Shape: (1, 3)
Columns: ['int_col', 'float_col', 'str_col']
DataFrame dtypes:
int_col        int64
float_col    float64
str_col       object
dtype: object
String column contains surrogate character: U+D800

Interchange object created successfully

Attempting to convert back using pd.api.interchange.from_dataframe()...

Error occurred: UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed

Full traceback:
```
</details>

## Why This Is A Bug

The DataFrame interchange protocol specification at https://data-apis.org/dataframe-protocol/latest/API.html explicitly states that string columns should be UTF-8 encoded (DtypeKind.STRING = 21 with UTF-8 encoding). However, pandas allows DataFrames to contain Python strings with surrogate characters (U+D800-U+DFFF) which are valid in Python's internal string representation but cannot be encoded to UTF-8.

This creates a mismatch where:
1. **Pandas accepts the data**: DataFrames can store strings with surrogates without any warnings or errors
2. **The interchange protocol requires UTF-8**: The protocol specification mandates UTF-8 encoding for string data
3. **The implementation crashes ungracefully**: Instead of handling the encoding error or providing a clear error message, the code raises an unhandled UnicodeEncodeError
4. **No documentation of this limitation**: Neither pandas documentation nor the interchange protocol documentation mentions this limitation or how to handle it

The bug violates the principle of least surprise - users reasonably expect that data which pandas accepts should be transferable through pandas' own interchange protocol, or at minimum receive a clear error message explaining why the operation cannot proceed.

## Relevant Context

The crash occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/column.py:351` in the `_get_data_buffer` method when encoding string data:

```python
# Line 349-351 in column.py
for obj in buf:
    if isinstance(obj, str):
        b.extend(obj.encode(encoding="utf-8"))  # <-- Crashes here
```

Surrogate characters appear in real-world scenarios:
- Data imported from databases with encoding issues
- Text processing where surrogates are used as placeholders
- Data from legacy systems with non-standard encoding
- Corrupted or partially decoded Unicode data

The pandas documentation at https://pandas.pydata.org/docs/reference/api/pandas.api.interchange.from_dataframe.html includes a warning that the implementation has known issues and recommends using Arrow for dataframe interchange instead.

## Proposed Fix

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,11 @@ class PandasColumn:
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    try:
+                        b.extend(obj.encode(encoding="utf-8"))
+                    except UnicodeEncodeError:
+                        raise ValueError(f"String contains surrogate characters (U+D800-U+DFFF) "
+                                       "which cannot be encoded to UTF-8. The interchange protocol "
+                                       "requires valid UTF-8 strings. Consider cleaning your data.")

             # Convert the byte array to a Pandas "buffer" using
```