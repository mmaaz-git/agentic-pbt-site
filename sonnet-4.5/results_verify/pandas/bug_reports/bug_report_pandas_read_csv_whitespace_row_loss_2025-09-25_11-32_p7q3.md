# Bug Report: pandas.io.parsers.read_csv Whitespace-Only Row Data Loss

**Target**: `pandas.io.parsers.read_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Rows containing only whitespace characters are silently dropped during CSV round-trip, causing data loss. This violates the round-trip property and can lead to silent data corruption.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st


@given(
    data=st.lists(
        st.lists(
            st.text(alphabet=' ', min_size=1, max_size=5),
            min_size=1,
            max_size=3
        ),
        min_size=1,
        max_size=10
    )
)
def test_roundtrip_whitespace_dataframe(data):
    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        return

    col_names = [f"col{i}" for i in range(num_cols)]
    df = pd.DataFrame(data, columns=col_names)

    csv_str = df.to_csv(index=False)
    df_roundtrip = pd.read_csv(io.StringIO(csv_str))

    assert len(df) == len(df_roundtrip), \
        f"Row count changed: {len(df)} -> {len(df_roundtrip)}"
```

**Failing input**: `data=[[' ']]`

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame([[' ']], columns=['col0'])
csv_str = df.to_csv(index=False)
df_roundtrip = pd.read_csv(io.StringIO(csv_str))

print(f"Original rows: {len(df)}")
print(f"After round-trip rows: {len(df_roundtrip)}")
print(f"CSV content: {repr(csv_str)}")
```

Output:
```
Original rows: 1
After round-trip rows: 0
CSV content: 'col0\n \n'
```

## Why This Is A Bug

This is a **data loss bug**. An entire row of data is silently dropped during CSV round-trip because it contains only whitespace. Whitespace-only values are legitimate data that should be preserved.

Users saving and loading DataFrames expect to get the same data back. Losing rows without any warning or error is a serious issue that can corrupt datasets and break data pipelines.

## Fix

The issue is caused by the default `skip_blank_lines=True` parameter in `read_csv`. A line containing only whitespace is treated as "blank" and skipped. However, `to_csv` writes the whitespace value to the CSV file, creating an asymmetry.

Workaround for users:
```python
df_roundtrip = pd.read_csv(io.StringIO(csv_str), skip_blank_lines=False)
```

This preserves whitespace-only rows, though users would not know to use this parameter without encountering the bug first.

A better solution would be to distinguish between truly blank lines (empty lines) and lines containing whitespace characters. Only truly empty lines should be skipped by default.