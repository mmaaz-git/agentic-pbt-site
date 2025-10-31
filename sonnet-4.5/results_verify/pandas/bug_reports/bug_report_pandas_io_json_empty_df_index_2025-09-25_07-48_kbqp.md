# Bug Report: pandas.io.json Empty DataFrame Index Type Not Preserved

**Target**: `pandas.io.json` (specifically `to_json` and `read_json` with `orient='split'`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When round-tripping an empty DataFrame through JSON with `orient='split'`, the index type is not preserved. A `RangeIndex` is converted to a regular `Index` with `float64` dtype.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import tempfile


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
def test_json_round_trip_orient_split(df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        df.to_json(temp_path, orient="split")
        result = pd.read_json(temp_path, orient="split")
        pd.testing.assert_frame_equal(df, result)
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
print(f"Original index type: {type(df.index)}")
print(f"Original index: {df.index}")

with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    temp_path = f.name

df.to_json(temp_path, orient="split")
result = pd.read_json(temp_path, orient="split")

print(f"Result index type: {type(result.index)}")
print(f"Result index: {result.index}")
print(f"Index types match: {type(df.index) == type(result.index)}")
```

Output:
```
Original index type: <class 'pandas.core.indexes.range.RangeIndex'>
Original index: RangeIndex(start=0, stop=0, step=1)
Result index type: <class 'pandas.core.indexes.base.Index'>
Result index: Index([], dtype='float64')
Index types match: False
```

## Why This Is A Bug

The `orient='split'` format is supposed to preserve the structure of the DataFrame for round-trip serialization. However, for empty DataFrames, the index type is not preserved. The original `RangeIndex(start=0, stop=0, step=1)` becomes a regular `Index([], dtype='float64')`. This violates the expected contract that `read_json(df.to_json(orient='split'), orient='split')` should be equivalent to the original DataFrame.

This affects users who:
1. Serialize empty DataFrames (common after filtering operations)
2. Expect the index type to be preserved for consistency
3. Rely on `RangeIndex` for memory efficiency or type checking

## Fix

The issue occurs because when the index is empty, the JSON serialization doesn't preserve the index type information. The fix would involve:

1. Including index type metadata in the JSON output for `orient='split'`
2. Reading and restoring the correct index type when deserializing

For the `orient='table'` format (which includes schema information), this works correctly. The `orient='split'` format should either:
- Include minimal type information for the index, or
- Document that index types may not be preserved for empty DataFrames