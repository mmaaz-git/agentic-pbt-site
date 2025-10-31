# Bug Report: pandas.io CSV Large Integer Type Corruption

**Target**: `pandas.io.parsers.read_csv` and `pandas.DataFrame.to_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame containing Python integers beyond int64 range is written to CSV and read back, the integer values are silently converted to strings, causing type corruption and comparison failures.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import io
from pandas.testing import assert_frame_equal

@given(
    st.lists(st.integers(), min_size=1, max_size=50),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1, max_size=50)
)
@settings(max_examples=100)
def test_csv_round_trip_basic(int_data, float_data):
    """Round-trip property: read_csv(to_csv(df)) should equal df (basic types)"""
    assume(len(int_data) == len(float_data))

    df = pd.DataFrame({'int_col': int_data, 'float_col': float_data})

    csv_str = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_str))

    assert len(result) == len(df)
    assert list(result.columns) == list(df.columns)
    assert_frame_equal(result, df)
```

**Failing input**: `int_data=[-9223372036854775809]` (value less than min int64: -2^63)

## Reproducing the Bug

```python
import pandas as pd
import io

int_value = -9223372036854775809

df = pd.DataFrame({'int_col': [int_value]})
print(f"Original: {df['int_col'].iloc[0]} (type: {type(df['int_col'].iloc[0]).__name__})")

csv_str = df.to_csv(index=False)
result = pd.read_csv(io.StringIO(csv_str))
print(f"After round-trip: {result['int_col'].iloc[0]} (type: {type(result['int_col'].iloc[0]).__name__})")

print(f"\nValues equal: {df['int_col'].iloc[0] == result['int_col'].iloc[0]}")
print(f"Types match: {type(df['int_col'].iloc[0]) == type(result['int_col'].iloc[0])}")
```

Output:
```
Original: -9223372036854775809 (type: int)
After round-trip: -9223372036854775809 (type: str)

Values equal: False
Types match: False
```

## Why This Is A Bug

1. **Silent type corruption**: The value changes from `int` to `str` without warning
2. **Comparison failures**: `int(-9223372036854775809) != str('-9223372036854775809')`
3. **Arithmetic failures**: String values cannot be used in mathematical operations
4. **Unexpected behavior**: Users expect CSV round-trip to preserve Python integer types, especially since pandas already stores the original value in an object dtype column

The bug occurs because:
- Original DataFrame stores the large int in object dtype (correct, as it exceeds int64)
- `to_csv()` writes the integer as a numeric string: `-9223372036854775809`
- `read_csv()` tries to infer the type from the CSV text
- The parser sees the value exceeds int64 range and falls back to object dtype
- However, it stores it as a **string** instead of parsing it back as a Python int

## Fix

The issue is in pandas' CSV type inference. When a numeric value in CSV exceeds int64 range, the parser should store it as a Python `int` object (like the original), not as a `str`.

In `pandas/io/parsers/c_parser_wrapper.py` or the relevant parser code, when integer parsing encounters overflow, it should:

```python
try:
    value = int(string_value)
except ValueError:
    value = string_value
```

Rather than immediately falling back to string dtype when the value exceeds int64 range.