# Bug Report: pandas.io.json Empty DataFrame Loses Columns

**Target**: `pandas.io.json` (specifically `to_json` and `read_json` with `orient='records'`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When serializing an empty DataFrame to JSON with `orient='records'`, all column information is lost. The resulting JSON is just `[]`, and when read back, pandas creates a completely empty DataFrame with no columns.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import tempfile
import os


@given(
    data_frames(
        columns=[
            column("int_col", dtype=int),
            column("float_col", dtype=float),
            column("str_col", dtype=str),
        ],
        index=range_indexes(min_size=0, max_size=100),
    )
)
@settings(max_examples=100)
def test_json_round_trip_orient_records(df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        df.to_json(temp_path, orient="records")
        result = pd.read_json(temp_path, orient="records")
        pd.testing.assert_frame_equal(df, result, check_index_type=False)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

**Failing input**: Empty DataFrame with columns `["int_col", "float_col", "str_col"]`

## Reproducing the Bug

```python
import pandas as pd
import tempfile

df = pd.DataFrame(columns=["int_col", "float_col", "str_col"])
print(f"Original columns: {df.columns.tolist()}")
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    temp_path = f.name

df.to_json(temp_path, orient="records")

with open(temp_path) as f:
    json_content = f.read()
print(f"JSON content: {json_content}")

result = pd.read_json(temp_path, orient="records")
print(f"Result columns: {result.columns.tolist()}")
print(f"Result shape: {result.shape}")
```

Output:
```
Original columns: ['int_col', 'float_col', 'str_col']
Original shape: (0, 3)
JSON content: []
Result columns: []
Result shape: (0, 0)
```

## Why This Is A Bug

Empty DataFrames with column metadata are extremely common in data processing pipelines (e.g., after filtering, before data arrives, in initialization). The `orient='records'` format completely loses this metadata, making round-trip serialization impossible.

This is a high-severity bug because:
1. **Silent data loss**: Column names and structure are completely lost
2. **Common use case**: Empty DataFrames frequently occur in real applications
3. **Violates expectations**: Users expect column information to be preserved
4. **Breaking change**: Code that works with non-empty DataFrames silently fails with empty ones

## Fix

The `orient='records'` format represents a DataFrame as an array of row objects. For an empty DataFrame, this becomes `[]`. When reading back `[]`, pandas has no way to infer the columns.

Possible fixes:

1. **Raise an error** when calling `to_json(orient='records')` on an empty DataFrame with a clear message directing users to use `orient='split'` or `orient='table'` instead.

2. **Use a special format** for empty DataFrames, such as:
   ```json
   [{"__pandas_empty_columns__": ["int_col", "float_col", "str_col"]}]
   ```

3. **Document the limitation** clearly in the docstring that `orient='records'` cannot preserve column information for empty DataFrames and recommend alternative orientations.

Option 1 (raising an error) is the safest approach to prevent silent data loss.