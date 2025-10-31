# Bug Report: pandas.io.parsers read_csv Large Integer Type Loss

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame containing Python integers larger than int64 range is written to CSV and read back using `read_csv`, the integers are incorrectly converted to strings instead of being parsed as integers.

## Property-Based Test

```python
import io
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    overflow_amount=st.integers(min_value=1, max_value=10**15)
)
@settings(max_examples=100)
def test_large_int_round_trip_preserves_type(overflow_amount):
    """
    Property: CSV round-trip should preserve numeric types, including integers
    that don't fit in int64.

    Evidence: DataFrames can store Python integers of arbitrary size (as object dtype),
    and CSV round-trip should preserve this data type, not convert to string.
    """
    int64_min = np.iinfo(np.int64).min
    overflow_value = int64_min - overflow_amount

    df = pd.DataFrame({'x': [overflow_value]})

    assert df['x'].dtype == object
    assert isinstance(df['x'].iloc[0], int)

    csv_string = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_string))

    assert isinstance(result['x'].iloc[0], int), \
        f"Expected int, got {type(result['x'].iloc[0]).__name__}"
    assert result['x'].iloc[0] == overflow_value
```

**Failing input**: `overflow_amount=1` (or any positive value)

## Reproducing the Bug

```python
import io
import pandas as pd
import numpy as np

int64_min = np.iinfo(np.int64).min
overflow_value = int64_min - 1

df = pd.DataFrame({'x': [overflow_value]})

csv_string = df.to_csv(index=False)
result = pd.read_csv(io.StringIO(csv_string))

print(f"Original type: {type(df['x'].iloc[0])}")
print(f"After CSV round-trip: {type(result['x'].iloc[0])}")

assert isinstance(df['x'].iloc[0], int)
assert isinstance(result['x'].iloc[0], str)
```

## Why This Is A Bug

1. **Data Type Inconsistency**: Pandas supports arbitrary-precision Python integers through object dtype. When such values are written to CSV, they should be read back as integers, not strings.

2. **Violates Round-Trip Property**: The fundamental expectation of serialization/deserialization is that data types should be preserved. This bug silently changes numeric data to string data.

3. **Silent Data Corruption**: Users expecting numeric operations on these values after reading will encounter type errors or unexpected behavior.

4. **Pandas Already Handles This Case**: The DataFrame correctly stores the large integer as object dtype and writes it correctly to CSV. Only the reading phase fails to parse it back as an integer.

## Fix

The issue is in the CSV parser's type inference logic. When a value in a column doesn't fit in int64, the parser should attempt to parse it as a Python `int` object rather than falling back to string.

A potential fix would be in the C parser wrapper or Python parser to:
1. Detect when an integer string exceeds int64 bounds
2. Parse it using Python's `int()` function
3. Store the result with object dtype

The fix would likely be in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/c_parser_wrapper.py` or similar parser implementation files, in the type conversion logic.