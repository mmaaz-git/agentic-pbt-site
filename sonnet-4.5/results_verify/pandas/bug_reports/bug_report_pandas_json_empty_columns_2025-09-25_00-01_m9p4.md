# Bug Report: pandas.io.json Empty DataFrame Loses Columns

**Target**: `pandas.io.json.read_json` / `pandas.io.json.to_json`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping an empty DataFrame with `orient='records'` or `orient='values'`, the column information is lost. The serialized JSON is `[]`, and when read back, pandas creates a (0, 0) DataFrame instead of preserving the original columns.

## Property-Based Test

```python
import pandas as pd
from io import StringIO
from hypothesis import given, settings, strategies as st
from pandas.testing import assert_frame_equal

@st.composite
def dataframes(draw):
    num_rows = draw(st.integers(min_value=0, max_value=20))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    columns = [f"col_{i}" for i in range(num_cols)]
    data = {col: draw(st.lists(st.integers(), min_size=num_rows, max_size=num_rows))
            for col in columns}
    return pd.DataFrame(data)

@given(dataframes())
@settings(max_examples=200)
def test_dataframe_roundtrip_records(df):
    json_str = df.to_json(orient='records')
    df_roundtrip = pd.read_json(StringIO(json_str), orient='records')
    assert_frame_equal(df.reset_index(drop=True), df_roundtrip, check_dtype=False)
```

**Failing input**: `pd.DataFrame({"col_0": []})`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({"col_0": []})

print(f"Original: shape={df.shape}, columns={df.columns.tolist()}")

json_str = df.to_json(orient='records')
print(f"JSON: {json_str}")

df_roundtrip = pd.read_json(StringIO(json_str), orient='records')
print(f"Roundtrip: shape={df_roundtrip.shape}, columns={df_roundtrip.columns.tolist()}")
```

Output:
```
Original: shape=(0, 1), columns=['col_0']
JSON: []
Roundtrip: shape=(0, 0), columns=[]
```

The same issue occurs with `orient='values'`:

```python
df = pd.DataFrame({"col_0": []})
json_str = df.to_json(orient='values')
df_roundtrip = pd.read_json(StringIO(json_str), orient='values')

print(f"Original: {df.shape}")
print(f"Roundtrip: {df_roundtrip.shape}")
```

Output:
```
Original: (0, 1)
Roundtrip: (0, 0)
```

## Why This Is A Bug

1. **Violates documented round-trip guarantee**: The documentation explicitly states that `read_json` can parse "compatible JSON strings" produced by `to_json()` with the corresponding orient value.

2. **Silently corrupts data structure**: Column names and schema information are critical metadata. Losing this information silently is a serious data integrity issue.

3. **Common use case**: Empty DataFrames are common when:
   - Filtering data yields no results
   - Initializing a schema before populating data
   - Working with partitioned data where some partitions are empty

4. **Inconsistent behavior**: Other orients like 'split' preserve columns (though they have the index type bug), making this behavior surprising.

## Fix

The root cause is that for `orient='records'` and `orient='values'`, an empty DataFrame serializes to `[]`, which contains no column information.

Possible solutions:

1. **For orient='records'**: Serialize empty DataFrame as `[]` but include a special marker or always use an object with column info:
   ```json
   {"columns": ["col_0"], "data": []}
   ```

2. **For orient='values'**: This orient fundamentally cannot preserve column names since it only contains values. The documentation should explicitly warn users that empty DataFrames cannot round-trip with this orient.

3. **Recommended approach**: For 'records' orient, when serializing an empty DataFrame, include the column schema either in a wrapper object or use 'split' orient as a fallback for empty DataFrames.