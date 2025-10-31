# Bug Report: pandas.io.json Float Dtype Lost on Round-Trip

**Target**: `pandas.io.json.read_json` and `pandas.DataFrame.to_json`
**Severity**: High
**Bug Type**: Logic

**Date**: 2025-09-25

## Summary

When a DataFrame with float columns containing integer-like values (0.0, 1.0, 2.0) is serialized with `to_json()` and deserialized with `read_json()`, the dtype changes from float64 to int64, causing silent data corruption.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1, max_size=50))
@settings(max_examples=100)
def test_json_round_trip_with_special_floats(floats_list):
    df = pd.DataFrame({"vals": floats_list})
    json_str = df.to_json()
    df_round_trip = pd.read_json(io.StringIO(json_str))
    pd.testing.assert_frame_equal(df, df_round_trip)
```

**Failing input**: `[0.0]` (or any list of floats that look like integers: `[1.0]`, `[2.0, 3.0]`, etc.)

## Reproducing the Bug

```python
import io
import pandas as pd

df = pd.DataFrame({"vals": [0.0, 1.0, 2.0]})
print(f"Original dtype: {df['vals'].dtype}")

json_str = df.to_json()
print(f"JSON: {json_str}")

df_round_trip = pd.read_json(io.StringIO(json_str))
print(f"Round-trip dtype: {df_round_trip['vals'].dtype}")

assert df['vals'].dtype == df_round_trip['vals'].dtype
```

Output:
```
Original dtype: float64
JSON: {"vals":{"0":0.0,"1":1.0,"2":2.0}}
Round-trip dtype: int64
AssertionError
```

## Why This Is A Bug

This is a high-severity bug that causes silent data corruption:

1. **Silent type coercion**: Float columns are converted to int without warning
2. **Loss of type information**: The JSON correctly contains float values (0.0, 1.0), but `read_json` incorrectly infers them as integers
3. **Inconsistent behavior**: Floats with fractional parts (0.5, 1.5) are preserved correctly, but integer-like floats are not
4. **Breaks round-trip property**: `read_json(df.to_json())` should equal `df`, but it doesn't when the DataFrame contains float columns with integer-like values
5. **Affects downstream code**: Code that relies on dtype (e.g., division operations, type checking) will behave differently

## Fix

The issue is in `read_json`'s dtype inference logic. When parsing numeric values, it should respect the JSON type (float) rather than converting based on the value's appearance.

Workarounds exist (`dtype=False` or `dtype={'vals': float}`), but the default behavior should preserve dtypes.

Proposed fix:
1. Modify the dtype inference in `read_json` to preserve float types from JSON
2. When a JSON value is explicitly a float (has decimal point), preserve it as float64 even if the value looks like an integer
3. Add a test case to ensure float dtypes are preserved in round-trip serialization