# Bug Report: pandas.api.interchange.from_dataframe UnicodeEncodeError on UTF-16 Surrogate Characters

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `from_dataframe` function crashes with `UnicodeEncodeError` when processing DataFrames containing string columns with UTF-16 surrogate characters (U+D800 to U+DFFF), violating the expectation that data accepted by pandas should be processable through the interchange protocol.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra.pandas import data_frames, column


@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float),
    column('c', dtype=str),
]))
@example(pd.DataFrame({'a': [1], 'b': [1.0], 'c': ['\ud800']}))  # Adding the failing example
@settings(max_examples=100)
def test_column_names_preserved(df):
    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    assert list(interchange_obj.column_names()) == list(result.columns)

if __name__ == "__main__":
    test_column_names_preserved()
```

<details>

<summary>
**Failing input**: `pd.DataFrame({'a': [1], 'b': [1.0], 'c': ['\ud800']})`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/8
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_column_names_preserved FAILED                              [100%]

=================================== FAILURES ===================================
_________________________ test_column_names_preserved __________________________
hypo.py:11: in test_column_names_preserved
    column('a', dtype=int),
               ^^^
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo.py:19: in test_column_names_preserved
    result = from_dataframe(interchange_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py:100: in from_dataframe
    return _from_dataframe(
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py:123: in _from_dataframe
    pandas_df = protocol_df_chunk_to_pandas(chunk)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py:175: in protocol_df_chunk_to_pandas
    columns[name], buf = string_column_to_ndarray(col)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py:292: in string_column_to_ndarray
    buffers = col.get_buffers()
              ^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py:287: in get_buffers
    "data": self._get_data_buffer(),
            ^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py:351: in _get_data_buffer
    b.extend(obj.encode(encoding="utf-8"))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
E   Falsifying explicit example: test_column_names_preserved(
E       df=
E              a    b  c
E           0  1  1.0  \ud800
E       ,
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_column_names_preserved - UnicodeEncodeError: 'utf-8' cod...
============================== 1 failed in 0.37s ===============================
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
import traceback

df = pd.DataFrame({'col': ['\ud800']})
print("Created DataFrame with surrogate character U+D800")
print()

try:
    interchange_obj = df.__dataframe__()
    print("Successfully created interchange object")
    print()

    print("Attempting from_dataframe conversion...")
    result = from_dataframe(interchange_obj)
    print("Conversion successful!")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}")
    print(f"Error message: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
```

<details>

<summary>
UnicodeEncodeError at line 351 in pandas/core/interchange/column.py
</summary>
```
Created DataFrame with surrogate character U+D800

Successfully created interchange object

Attempting from_dataframe conversion...
Error occurred: UnicodeEncodeError
Error message: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed

Full traceback:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/repo.py", line 18, in <module>
    result = from_dataframe(interchange_obj)
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

UTF-16 surrogate characters (code points U+D800 through U+DFFF) are isolated code units that were designed for UTF-16 encoding but cannot be encoded in UTF-8 by specification. While these characters are technically invalid Unicode when appearing in isolation, Python strings and pandas DataFrames accept them without error. This creates an inconsistency where:

1. **pandas accepts the data**: `pd.DataFrame({'col': ['\ud800']})` creates a valid DataFrame without warnings or errors
2. **The interchange protocol fails**: The same data cannot be processed through `from_dataframe(df.__dataframe__())`
3. **The failure is ungraceful**: Instead of providing a clear error message about unsupported characters, the code crashes with an unhandled `UnicodeEncodeError`
4. **The limitation is undocumented**: Neither the pandas documentation nor the interchange protocol documentation mentions this limitation with surrogate characters

The DataFrame interchange protocol specification explicitly requires UTF-8 encoding for string data (DtypeKind.STRING = 21, UTF-8), but does not specify how to handle edge cases where UTF-8 encoding is impossible. This violates the principle of least surprise - users reasonably expect that data accepted by pandas should be processable through pandas' own interchange protocol.

## Relevant Context

The crash occurs in `/pandas/core/interchange/column.py` at line 351 where the code attempts to encode strings as UTF-8:
```python
for obj in buf:
    if isinstance(obj, str):
        b.extend(obj.encode(encoding="utf-8"))  # Line 351 - crashes here
```

The DataFrame interchange protocol is defined by the Python dataframe API standard (https://data-apis.org/dataframe-protocol/latest/), which specifies that string columns must use UTF-8 encoding. However, the implementation doesn't handle the edge case where Python strings contain characters that cannot be represented in UTF-8.

While surrogate characters are rare in practice, they can appear in data from various sources:
- Data corrupted during encoding/decoding processes
- Legacy systems using non-standard encodings
- Test data generators (like Hypothesis) that explore edge cases
- Malformed data from external sources

## Proposed Fix

The most robust solution is to handle the encoding error gracefully and provide fallback behavior:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,14 @@ class PandasColumn:
         # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
         for obj in buf:
             if isinstance(obj, str):
-                b.extend(obj.encode(encoding="utf-8"))
+                try:
+                    b.extend(obj.encode(encoding="utf-8"))
+                except UnicodeEncodeError:
+                    # Handle surrogate characters and other unencodable strings
+                    # Option 1: Use surrogatepass (platform-dependent)
+                    # Option 2: Replace with replacement character
+                    # Option 3: Use backslashreplace for reversibility
+                    b.extend(obj.encode(encoding="utf-8", errors="backslashreplace"))
```