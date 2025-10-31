# Bug Report: pandas.io.json Large Float Becomes Infinity

**Target**: `pandas.io.json` (ujson_dumps/ujson_loads round-trip)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping large (but valid finite) floats through `to_json`/`read_json`, the values are silently corrupted and become infinity. This happens because `ujson_dumps` rounds the float in a way that causes `ujson_loads` to parse it as infinity.

## Property-Based Test

```python
from io import StringIO
import pandas as pd
from hypothesis import given, settings, strategies as st


@given(
    st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=500)
def test_dataframe_roundtrip_columns(data):
    num_cols = len(data[0])
    assume(all(len(row) == num_cols for row in data))

    df = pd.DataFrame(data)
    assume(df.columns.is_unique)

    json_str = df.to_json(orient='columns')
    df_recovered = pd.read_json(StringIO(json_str), orient='columns')

    pd.testing.assert_frame_equal(df, df_recovered, check_dtype=False)
```

**Failing input**: `data=[[1.7976931345e+308]]`

## Reproducing the Bug

```python
from io import StringIO
import pandas as pd

df = pd.DataFrame([[1.7976931345e+308]])
print(f"Original value: {df.iloc[0, 0]}")
print(f"Is finite: {df.iloc[0, 0] != float('inf')}")

json_str = df.to_json(orient='columns')
print(f"JSON: {json_str}")

df_recovered = pd.read_json(StringIO(json_str), orient='columns')
print(f"Recovered value: {df_recovered.iloc[0, 0]}")
print(f"Is infinite: {df_recovered.iloc[0, 0] == float('inf')}")
```

Output:
```
Original value: 1.7976931345e+308
Is finite: True
JSON: {"0":{"0":1.797693135e+308}}
Recovered value: inf
Is infinite: True
```

## Why This Is A Bug

This is silent data corruption. A valid finite float `1.7976931345e+308` is converted to infinity after a round-trip through JSON serialization. The value is less than `sys.float_info.max` (1.7976931348623157e+308) and should remain finite.

The root cause is in the ujson library:
1. `ujson_dumps` rounds `1.7976931345e+308` to `1.797693135e+308`
2. `ujson_loads` incorrectly parses `1.797693135e+308` as infinity

The stdlib `json` module handles this correctly and preserves the value as finite.

## Fix

This is a bug in the ujson C library that pandas depends on. A workaround would be to use stdlib json for edge cases, or to validate that serialized floats can be deserialized without becoming infinity. The proper fix requires updating the ujson library to handle near-maximum floats correctly.