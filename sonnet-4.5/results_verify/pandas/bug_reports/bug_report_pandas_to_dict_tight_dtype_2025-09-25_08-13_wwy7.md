# Bug Report: pandas DataFrame to_dict/from_dict Dtype Loss

**Target**: `pandas.DataFrame.to_dict` / `pandas.DataFrame.from_dict`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Empty DataFrames lose dtype information when round-tripped through `to_dict(orient='tight')` and `from_dict(..., orient='tight')`, converting all columns to object dtype.

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
    assert result.equals(df), "Round-trip with orient='tight' should preserve DataFrame"
```

**Failing input**: Empty DataFrame with columns `[a, b]` where `a` has dtype `int64` and `b` has dtype `float64`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original dtypes:", df.dtypes.to_dict())

tight_dict = df.to_dict(orient='tight')
reconstructed = pd.DataFrame.from_dict(tight_dict, orient='tight')

print("Reconstructed dtypes:", reconstructed.dtypes.to_dict())
print("Equal?", df.equals(reconstructed))
```

Output:
```
Original dtypes: {'a': dtype('int64'), 'b': dtype('float64')}
Reconstructed dtypes: {'a': dtype('O'), 'b': dtype('O')}
Equal? False
```

## Why This Is A Bug

The `to_dict(orient='tight')` documentation states it creates a dictionary representation of the DataFrame, and `from_dict(..., orient='tight')` should reconstruct it. However, for empty DataFrames, dtype information is lost during the round-trip. This violates the fundamental expectation that serialization preserves data structure, especially since the `tight` format explicitly includes metadata fields like `index_names` and `column_names` suggesting it's designed to preserve full DataFrame structure.

This affects users who serialize empty DataFrames with specific schemas (e.g., for initialization, caching, or IPC), as the dtype information is critical for maintaining data contracts.

## Fix

The `tight` orientation dict should include dtype information. When `to_dict(orient='tight')` is called, it should add a `dtypes` field:

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
```

And `from_dict(..., orient='tight')` should restore dtypes when present:

```diff
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -1920,6 +1920,10 @@ class DataFrame(NDFrame):
             df = cls(data["data"], index=data["index"], columns=data["columns"])
             df.index.names = data.get("index_names")
             df.columns.names = data.get("column_names")
+            if "dtypes" in data:
+                for col, dtype in data["dtypes"].items():
+                    if col in df.columns:
+                        df[col] = df[col].astype(dtype)
             return df
```