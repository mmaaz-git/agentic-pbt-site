# Bug Report: pandas DataFrame to_dict/from_dict Dtype Loss for Empty DataFrames

**Target**: `pandas.DataFrame.to_dict` / `pandas.DataFrame.from_dict`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Empty DataFrames lose dtype information when round-tripped through `to_dict(orient='tight')` and `from_dict(..., orient='tight')`, converting all columns to object dtype instead of preserving their original types.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float)
]))
@settings(max_examples=200)
def test_to_dict_tight_from_dict_tight_roundtrip(df):
    dict_repr = df.to_dict(orient='tight')
    result = pd.DataFrame.from_dict(dict_repr, orient='tight')
    assert result.equals(df), f"Round-trip with orient='tight' failed. Original dtypes: {df.dtypes.to_dict()}, Result dtypes: {result.dtypes.to_dict()}"

if __name__ == "__main__":
    test_to_dict_tight_from_dict_tight_roundtrip()
```

<details>

<summary>
**Failing input**: Empty DataFrame with columns `[a, b]` where `a` has dtype `int64` and `b` has dtype `float64`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 16, in <module>
    test_to_dict_tight_from_dict_tight_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 6, in test_to_dict_tight_from_dict_tight_roundtrip
    column('a', dtype=int),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 13, in test_to_dict_tight_from_dict_tight_roundtrip
    assert result.equals(df), f"Round-trip with orient='tight' failed. Original dtypes: {df.dtypes.to_dict()}, Result dtypes: {result.dtypes.to_dict()}"
           ~~~~~~~~~~~~~^^^^
AssertionError: Round-trip with orient='tight' failed. Original dtypes: {'a': dtype('int64'), 'b': dtype('float64')}, Result dtypes: {'a': dtype('O'), 'b': dtype('O')}
Falsifying example: test_to_dict_tight_from_dict_tight_roundtrip(
    df=
        Empty DataFrame
        Columns: [a, b]
        Index: []
    ,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:345
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/__init__.py:43
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/common.py:127
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/common.py:128
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/common.py:139
        (and 80 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original DataFrame:")
print("  Columns:", df.columns.tolist())
print("  Dtypes:", df.dtypes.to_dict())
print("  Shape:", df.shape)
print()

tight_dict = df.to_dict(orient='tight')
print("Dictionary representation (orient='tight'):")
print("  Keys:", list(tight_dict.keys()))
print("  Dict:", tight_dict)
print()

reconstructed = pd.DataFrame.from_dict(tight_dict, orient='tight')
print("Reconstructed DataFrame:")
print("  Columns:", reconstructed.columns.tolist())
print("  Dtypes:", reconstructed.dtypes.to_dict())
print("  Shape:", reconstructed.shape)
print()

print("DataFrames equal?", df.equals(reconstructed))
print()

if not df.equals(reconstructed):
    print("ERROR: Round-trip through to_dict/from_dict with orient='tight' lost dtype information!")
    print("  Expected dtypes:", df.dtypes.to_dict())
    print("  Got dtypes:", reconstructed.dtypes.to_dict())
```

<details>

<summary>
ERROR: DataFrame round-trip loses dtype information for empty DataFrames
</summary>
```
Original DataFrame:
  Columns: ['a', 'b']
  Dtypes: {'a': dtype('int64'), 'b': dtype('float64')}
  Shape: (0, 2)

Dictionary representation (orient='tight'):
  Keys: ['index', 'columns', 'data', 'index_names', 'column_names']
  Dict: {'index': [], 'columns': ['a', 'b'], 'data': [], 'index_names': [None], 'column_names': [None]}

Reconstructed DataFrame:
  Columns: ['a', 'b']
  Dtypes: {'a': dtype('O'), 'b': dtype('O')}
  Shape: (0, 2)

DataFrames equal? False

ERROR: Round-trip through to_dict/from_dict with orient='tight' lost dtype information!
  Expected dtypes: {'a': dtype('int64'), 'b': dtype('float64')}
  Got dtypes: {'a': dtype('O'), 'b': dtype('O')}
```
</details>

## Why This Is A Bug

This behavior violates the fundamental expectation that serialization preserves data structure, especially for the 'tight' orientation which is specifically designed to preserve "full structural information" about the DataFrame.

The bug manifests only with empty DataFrames - non-empty DataFrames correctly preserve their dtypes through the same round-trip operation. This inconsistency is problematic because:

1. **Breaks data contracts**: Empty DataFrames with specific schemas are commonly used to establish data contracts between components. When dtype information is lost, subsequent operations may fail or produce incorrect results.

2. **Performance implications**: Converting numeric columns to object dtype causes significant performance degradation for subsequent operations.

3. **Inconsistent behavior**: The same operation preserves dtypes for non-empty DataFrames but not for empty ones, violating the principle of least surprise.

4. **Documentation mismatch**: The 'tight' format includes metadata fields like `index_names` and `column_names`, suggesting it's designed for complete DataFrame reconstruction. The documentation states it preserves "full structural information", which users reasonably interpret to include data types.

## Relevant Context

The 'tight' orientation was introduced in pandas 1.4.0 as a more complete serialization format that includes metadata about index and column names. The dictionary structure currently includes:
- `index`: the index values
- `columns`: the column names
- `data`: the data values
- `index_names`: names of the index levels
- `column_names`: names of the column levels

However, it notably omits dtype information, which causes pandas to fall back to default type inference when reconstructing the DataFrame. For empty data arrays, pandas defaults all columns to object dtype since it cannot infer types from the values.

This issue affects common use cases such as:
- Template DataFrames with predefined schemas
- Accumulator DataFrames initialized as empty
- Schema validation and communication between processes
- Caching of empty query results with correct types

The pandas documentation for `from_dict` does provide a `dtype` parameter, but this applies a single dtype to all columns and cannot preserve per-column dtype information.

## Proposed Fix

The fix requires modifying both `to_dict` and `from_dict` methods to include and restore dtype information in the 'tight' format:

```diff
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -1850,6 +1850,7 @@ class DataFrame(NDFrame):
             "columns": self.columns.tolist(),
             "data": self.values.tolist(),
             "index_names": list(self.index.names),
             "column_names": list(self.columns.names),
+            "dtypes": {col: str(dtype) for col, dtype in self.dtypes.items()},
         }

@@ -1920,6 +1921,11 @@ class DataFrame(NDFrame):
             df = cls(data["data"], index=data["index"], columns=data["columns"])
             df.index.names = data.get("index_names")
             df.columns.names = data.get("column_names")
+            if "dtypes" in data:
+                for col, dtype_str in data["dtypes"].items():
+                    if col in df.columns:
+                        import numpy as np
+                        df[col] = df[col].astype(np.dtype(dtype_str))
             return df
```