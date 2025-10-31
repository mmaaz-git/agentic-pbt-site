# Bug Report: pandas.io.json Round-Trip Converts Numeric Values to Datetime

**Target**: `pandas.io.json.read_json` / `pandas.Series.to_json`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping a pandas Series containing numeric (float64) values through JSON serialization, `read_json` automatically converts values greater than 31536000 to datetime objects, violating the round-trip property and silently changing both the dtype and values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from io import StringIO
import pandas as pd


@given(st.floats(min_value=31536001, max_value=1e10, allow_nan=False, allow_infinity=False))
def test_series_json_roundtrip_preserves_dtype_and_value(value):
    series = pd.Series([value])

    json_str = series.to_json(orient="split")
    result = pd.read_json(StringIO(json_str), typ="series", orient="split")

    assert series.dtype == result.dtype, f"dtype changed from {series.dtype} to {result.dtype}"
    assert series.iloc[0] == result.iloc[0], f"value changed from {series.iloc[0]} to {result.iloc[0]}"
```

**Failing input**: `31536001.0` (or any float >= 31536001)

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

series = pd.Series([31536001.0])
print(f"Original: {series.iloc[0]} (dtype: {series.dtype})")

json_str = series.to_json(orient="split")
result = pd.read_json(StringIO(json_str), typ="series", orient="split")

print(f"After round-trip: {result.iloc[0]} (dtype: {result.dtype})")
```

Output:
```
Original: 31536001.0 (dtype: float64)
After round-trip: 1971-01-01 00:00:01 (dtype: datetime64[ns])
```

Additional examples:
```python
pd.Series([31536000.0]).to_json(orient="split") -> dtype changes from float64 to int64
pd.Series([1000000000.5]).to_json(orient="split") -> value becomes 2001-09-09 01:46:40.500000
```

## Why This Is A Bug

The round-trip property `read_json(series.to_json(...), ...) ≈ series` is a fundamental expectation for serialization. The automatic date conversion in `read_json` is too aggressive because:

1. **Silent data corruption**: Numeric values are silently converted to datetime objects without user consent
2. **Type instability**: The dtype changes from float64 to datetime64[ns], breaking type contracts
3. **Loss of information**: Values like 1000000000.5 lose their numeric interpretation
4. **Inconsistent behavior**: The threshold (31536000) is arbitrary from a user perspective who just wants to serialize numbers

The docstring for `read_json` shows examples of round-trip serialization (lines 710-743 in _json.py), implying this should work correctly. The automatic conversion violates this implied contract.

## Fix

The root cause is in `pandas/io/json/_json.py`, lines 1329-1337 in the `_try_convert_to_date` method:

```python
if issubclass(new_data.dtype.type, np.number):
    in_range = (
        isna(new_data._values)
        | (new_data > self.min_stamp)
        | (new_data._values == iNaT)
    )
    if not in_range.all():
        return data, False
```

**Recommended fix**: Make the automatic date conversion opt-in rather than opt-out. Change the default value of `convert_dates` to `False` for numeric data, or only apply date conversion when:
1. The column name suggests it's a date (already handled by `keep_default_dates`)
2. The user explicitly passes `convert_dates=True` or a list of columns
3. Metadata in the JSON indicates the original type was datetime

**Minimal fix**: Document this behavior clearly and provide a warning when numeric data is auto-converted to datetime, so users can add `convert_dates=False` to their `read_json` calls.

**Workaround**: Users can avoid this bug by explicitly passing `convert_dates=False` to `read_json`:

```python
result = pd.read_json(StringIO(json_str), typ="series", orient="split", convert_dates=False)
```

However, this should not be necessary for basic round-trip serialization.