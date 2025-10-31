# Bug Report: pandas.core.interchange.column Unicode Surrogate Encoding Crash

**Target**: `pandas.core.interchange.column.PandasColumn._get_data_buffer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas interchange protocol crashes with a `UnicodeEncodeError` when processing DataFrames containing Unicode surrogate characters (U+D800-U+DFFF), preventing data exchange between pandas instances.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.interchange.from_dataframe import from_dataframe

@given(st.data())
@settings(max_examples=200)
def test_from_dataframe_with_unicode_strings(data):
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    values = []
    for _ in range(n_rows):
        val = data.draw(st.text(
            alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0x1FFFF),
            min_size=0,
            max_size=20
        ))
        values.append(val)

    df_original = pd.DataFrame({'col': values})
    interchange_obj = df_original.__dataframe__()
    df_roundtrip = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(df_original, df_roundtrip)
```

<details>

<summary>
**Failing input**: `'\ud800'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 26, in <module>
    test_from_dataframe_with_unicode_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 6, in test_from_dataframe_with_unicode_strings
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 21, in test_from_dataframe_with_unicode_strings
    df_roundtrip = from_dataframe(interchange_obj)
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
Falsifying example: test_from_dataframe_with_unicode_strings(
    data=data(...),
)
Draw 1: 1
Draw 2: '\ud800'
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe
import traceback

# Create a DataFrame with a Unicode surrogate character
df = pd.DataFrame({'col': ['\ud800']})
print("Original DataFrame created")
print(f"Shape: {df.shape}")
print(f"String value representation: {repr(df['col'][0])}")

# Try to convert through interchange protocol
try:
    interchange_obj = df.__dataframe__()
    print("Interchange object created successfully")
    df_roundtrip = from_dataframe(interchange_obj)
    print("Roundtrip succeeded:")
    print(f"Roundtrip shape: {df_roundtrip.shape}")
except Exception as e:
    print(f"\nError during roundtrip conversion:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
UnicodeEncodeError during interchange roundtrip
</summary>
```
Original DataFrame created
Shape: (1, 1)
String value representation: '\ud800'
Interchange object created successfully

Error during roundtrip conversion:
Error type: UnicodeEncodeError
Error message: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed

Full traceback:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 15, in <module>
    df_roundtrip = from_dataframe(interchange_obj)
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

This violates the expected behavior of pandas' interchange protocol in several ways:

1. **Data Loss**: Valid pandas DataFrames that successfully store Unicode surrogate characters cannot be exchanged via the interchange protocol, causing data to become "trapped" in the original DataFrame.

2. **Inconsistent Handling**: Pandas accepts and stores strings containing surrogate characters without any issues (`pd.DataFrame({'col': ['\ud800']})` works fine), but the interchange protocol fails to handle them, creating an inconsistency in the library's Unicode handling.

3. **Unhandled Exception**: The code crashes with an unhandled `UnicodeEncodeError` rather than gracefully handling the edge case or providing a clear error message about data limitations.

4. **Violates Round-trip Expectation**: The interchange protocol's fundamental purpose is to enable data exchange. Users reasonably expect that `from_dataframe(df.__dataframe__())` should preserve data that pandas itself accepts.

5. **UTF-8 Standard vs Python Reality**: While UTF-8 technically doesn't support surrogate characters, Python strings can contain them, and Python's standard library provides the `surrogatepass` error handler specifically for such cases.

## Relevant Context

Unicode surrogate characters (U+D800 to U+DFFF) are code points reserved for UTF-16 encoding. While they're technically invalid in UTF-8, they can appear in real-world data from various sources:
- Legacy system migrations
- Data corruption or encoding issues
- Malformed user input
- Certain text processing operations

The crash occurs in two locations within `pandas/core/interchange/column.py`:
- Line 351 in `_get_data_buffer()`: `b.extend(obj.encode(encoding="utf-8"))`
- Line 439 in `_get_offsets_buffer()`: `b = v.encode(encoding="utf-8")`

Python provides the `surrogatepass` error handler specifically to handle this case, which allows encoding and decoding of surrogate characters for situations where data preservation is more important than strict UTF-8 compliance.

Documentation: https://docs.python.org/3/library/codecs.html#error-handlers
Code location: https://github.com/pandas-dev/pandas/blob/main/pandas/core/interchange/column.py

## Proposed Fix

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn:
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))

             # Convert the byte array to a Pandas "buffer" using
             # a NumPy array as the backing store
@@ -436,7 +436,7 @@ class PandasColumn:
                 # For missing values (in this case, `np.nan` values)
                 # we don't increment the pointer
                 if isinstance(v, str):
-                    b = v.encode(encoding="utf-8")
+                    b = v.encode(encoding="utf-8", errors="surrogatepass")
                     ptr += len(b)

                 offsets[i + 1] = ptr
```