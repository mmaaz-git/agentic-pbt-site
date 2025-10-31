# Bug Report: pandas.core.interchange Crashes on Strings with Lone Surrogates

**Target**: `pandas.core.interchange.column.PandasColumn._get_data_buffer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas interchange protocol crashes with a `UnicodeEncodeError` when converting DataFrames containing strings with lone surrogate characters (Unicode code points U+D800 to U+DFFF that are not part of a valid surrogate pair).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pytest


@given(st.text(alphabet=st.sampled_from(['\ud800', '\udc00', '\udfff']), min_size=1, max_size=10))
def test_string_with_surrogates_crashes(surrogate_str):
    df = pd.DataFrame({"str_col": [surrogate_str]})
    interchange_obj = df.__dataframe__()

    with pytest.raises(UnicodeEncodeError):
        pd.api.interchange.from_dataframe(interchange_obj)

# Run the test
if __name__ == "__main__":
    test_string_with_surrogates_crashes()
```

<details>

<summary>
**Failing input**: `'\ud800'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/1
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_string_with_surrogates_crashes PASSED                      [100%]

============================== 1 passed in 0.39s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({"str_col": ["\ud800"]})
interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)
```

<details>

<summary>
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/repo.py", line 5, in <module>
    result = pd.api.interchange.from_dataframe(interchange_obj)
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

This is a bug because pandas exhibits inconsistent behavior with lone surrogate characters:

1. **Pandas successfully stores these strings**: DataFrames can be created with and retrieve strings containing lone surrogates without any issues. The string type in Python allows these characters.

2. **The error occurs deep in internal code**: The crash happens at line 351 in `_get_data_buffer()` without any user-friendly error handling or validation at the API boundary.

3. **No documentation of this limitation**: Neither the pandas interchange documentation nor the DataFrame interchange protocol specification explicitly states that strings must be valid UTF-8 without surrogates.

4. **Poor user experience**: Users receive a cryptic error message from deep within the implementation rather than a clear explanation of the issue.

5. **The interchange protocol specification is vague**: While it mentions UTF-8 encoding for strings, it doesn't specify how implementations should handle edge cases like surrogates.

## Relevant Context

Testing reveals that pandas allows storing lone surrogates internally but fails on multiple export operations:

- **Storage works**: `pd.DataFrame({'str_col': ['\ud800']})` creates successfully
- **Interchange export fails**: Crashes with UnicodeEncodeError
- **CSV export fails**: `df.to_csv()` also crashes with the same error
- **JSON export fails**: `df.to_json()` crashes similarly

This shows a systemic issue with how pandas handles UTF-8 encoding for strings containing surrogates. While lone surrogates are technically invalid UTF-8, they are valid Python strings and can occur in real-world data from files with broken encodings or when processing certain binary data as text.

The crash location is in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/column.py:351` and also appears at line 439 in the same file for the offset calculation.

## Proposed Fix

Use the `surrogatepass` error handler to allow surrogates to be encoded and later decoded correctly:

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