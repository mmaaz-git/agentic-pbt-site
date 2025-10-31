# Bug Report: pandas CSV Round-Trip Data Corruption with Tab Character in Column Name

**Target**: `pandas.DataFrame.to_csv` / `pandas.read_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame column name contains only a tab character (`'\t'`), CSV round-trip silently corrupts data: the data value becomes the column name, and all actual data is lost.

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=10))
@settings(max_examples=200)
def test_csv_handles_special_chars_in_column_names(name):
    df = pd.DataFrame([[1]], columns=[name])
    csv_str = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_str))

    assert len(result.columns) == 1
    assert result.columns[0] == name
```

**Failing input**: `'\t'`

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame([[42]], columns=['\t'])
print("Original:")
print(f"  Columns: {list(df.columns)}")
print(f"  Values: {df.values.tolist()}")

csv_str = df.to_csv(index=False)
result = pd.read_csv(io.StringIO(csv_str))

print("\nAfter round-trip:")
print(f"  Columns: {list(result.columns)}")
print(f"  Values: {result.values.tolist()}")
```

**Output:**
```
Original:
  Columns: ['\t']
  Values: [[42]]

After round-trip:
  Columns: ['42']
  Values: []
```

## Why This Is A Bug

This violates the expected behavior of CSV round-trip:
1. The data value (42) becomes the column name
2. All actual data is lost (empty DataFrame)
3. The original column name (tab character) is completely lost
4. This is **silent data corruption** - no error or warning is raised

The CSV format interprets the tab in the header row as a delimiter, causing `read_csv` to misparse the structure.

## Fix

The `to_csv` function should properly escape or quote column names containing delimiter characters (tabs, commas, etc.) to prevent them from being interpreted as delimiters when reading back. The CSV writer should use quoting for column names that contain special characters.