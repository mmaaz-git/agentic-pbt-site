# Bug Report: pandas.api.interchange Surrogate Character Encoding Crash

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The interchange protocol crashes with a `UnicodeEncodeError` when attempting to convert pandas DataFrames containing surrogate characters (U+D800 through U+DFFF) back to pandas through `from_dataframe()`.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column
import pandas as pd
import pandas.api.interchange as interchange


@given(data_frames([
    column('A', dtype=int),
    column('B', dtype=float),
    column('C', dtype=str),
]))
@settings(max_examples=200)
def test_from_dataframe_round_trip(df):
    """
    Property: from_dataframe(df.__dataframe__()) should equal df for pandas DataFrames
    Evidence: The docstring shows this is the intended usage pattern
    """
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)
    pd.testing.assert_frame_equal(result, df)

# Run the test
if __name__ == "__main__":
    test_from_dataframe_round_trip()
```

<details>

<summary>
**Failing input**: `df=DataFrame({'A': [0], 'B': [0.0], 'C': ['\ud800']})`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 24, in <module>
    test_from_dataframe_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 8, in test_from_dataframe_round_trip
    column('A', dtype=int),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 19, in test_from_dataframe_round_trip
    result = interchange.from_dataframe(interchange_obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 100, in from_dataframe
    return _from_dataframe(
        df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 123, in _from_dataframe
    pandas_df = protocol_df_chunk_to_pandas(chunk)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 175, in protocol_df_chunk_to_pandas
    columns[name], buf = string_column_to_ndarray(col)
                         ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 292, in string_column_to_ndarray
    buffers = col.get_buffers()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 287, in get_buffers
    "data": self._get_data_buffer(),
            ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 351, in _get_data_buffer
    b.extend(obj.encode(encoding="utf-8"))
             ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
Falsifying example: test_from_dataframe_round_trip(
    df=
           A    B  C
        0  0  0.0  \ud800
    ,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.interchange as interchange

# Create a DataFrame with a surrogate character in the string column
df = pd.DataFrame({'A': [0], 'B': [0.0], 'C': ['\ud800']})

print("Created DataFrame with surrogate character U+D800 in column C")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame dtypes:\n{df.dtypes}")

# Try to convert through the interchange protocol
try:
    print("\nAttempting to convert through interchange protocol...")
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)
    print("Success! Result:")
    print(f"Result shape: {result.shape}")
    print(f"Result dtypes:\n{result.dtypes}")
except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
</summary>
```
Created DataFrame with surrogate character U+D800 in column C
DataFrame shape: (1, 3)
DataFrame dtypes:
A      int64
B    float64
C     object
dtype: object

Attempting to convert through interchange protocol...

Error occurred: UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/repo.py", line 15, in <module>
    result = interchange.from_dataframe(interchange_obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 100, in from_dataframe
    return _from_dataframe(
        df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 123, in _from_dataframe
    pandas_df = protocol_df_chunk_to_pandas(chunk)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 175, in protocol_df_chunk_to_pandas
    columns[name], buf = string_column_to_ndarray(col)
                         ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py", line 292, in string_column_to_ndarray
    buffers = col.get_buffers()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 287, in get_buffers
    "data": self._get_data_buffer(),
            ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py", line 351, in _get_data_buffer
    b.extend(obj.encode(encoding="utf-8"))
             ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Pandas accepts surrogate characters but interchange protocol rejects them**: pandas DataFrames allow surrogate characters (U+D800 through U+DFFF) in string columns without any warnings or errors. These DataFrames are considered valid by pandas, can be saved to CSV, pickled, and manipulated using all standard pandas operations. However, the interchange protocol crashes when attempting to process these valid DataFrames.

2. **Round-trip property violation**: The dataframe interchange protocol is designed to enable conversion between different DataFrame implementations. The documentation explicitly shows `from_dataframe(df.__dataframe__())` as the intended usage pattern for round-trip conversion. This pattern fails for DataFrames containing surrogates, breaking the fundamental contract of the protocol.

3. **UTF-8 specification conflict**: While UTF-8 (as defined in RFC 3629) explicitly forbids encoding surrogate characters, pandas has made the design decision to support them in DataFrames. The interchange protocol's strict UTF-8 requirement creates an incompatibility with pandas' more permissive string handling.

4. **Poor error handling**: The code directly calls `str.encode('utf-8')` without any error handling, resulting in an unhelpful crash deep in internal code (`pandas/core/interchange/column.py:351`) rather than providing a clear error message at the API boundary or gracefully handling the edge case.

5. **Undocumented limitation**: Neither the pandas documentation nor the interchange protocol documentation mentions that strings with surrogate characters will fail. Users have no way to know about this limitation until they encounter the crash.

6. **Real-world impact**: Surrogate characters can appear in real-world data from various sources including:
   - Corrupted text files or databases
   - Legacy Windows file names
   - Data imported from systems using UTF-16 encoding
   - Web scraping results with malformed HTML entities
   - Log files with encoding errors

## Relevant Context

The crash occurs in `pandas/core/interchange/column.py` at line 351 in the `_get_data_buffer()` method when processing string columns. The code iterates through string values and attempts to encode them as UTF-8:

```python
for obj in buf:
    if isinstance(obj, str):
        b.extend(obj.encode(encoding="utf-8"))  # Line 351 - crashes here
```

The interchange protocol specification (from data-apis.org) defines strings as `DtypeKind.STRING = 21  # UTF-8` but doesn't specify how implementations should handle invalid UTF-8 sequences like surrogates.

Python provides standard mechanisms for handling this exact situation through the `errors` parameter in the `encode()` method. The `surrogatepass` error handler specifically exists to allow surrogate characters to pass through encoding/decoding operations.

## Proposed Fix

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn(Column):
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))

             # Convert the byte array to a Pandas "buffer" using
             # a NumPy array as the backing store
@@ -436,7 +436,7 @@ class PandasColumn(Column):
                 # For missing values (in this case, `np.nan` values)
                 # we don't increment the pointer
                 if isinstance(v, str):
-                    b = v.encode(encoding="utf-8")
+                    b = v.encode(encoding="utf-8", errors="surrogatepass")
                     ptr += len(b)

                 offsets[i + 1] = ptr
```