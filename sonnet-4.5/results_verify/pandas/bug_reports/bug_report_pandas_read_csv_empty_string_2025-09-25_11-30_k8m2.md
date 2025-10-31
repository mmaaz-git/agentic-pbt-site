# Bug Report: pandas.io.parsers.read_csv Empty String to NaN Conversion

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property `read_csv(df.to_csv()) == df` is violated when the DataFrame contains empty strings. Empty strings are incorrectly converted to NaN values during CSV reading, causing data corruption.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st


@given(
    data=st.lists(
        st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=20), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    )
)
def test_roundtrip_string_dataframe(data):
    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        return

    for row in data:
        for val in row:
            if '\n' in val or '\r' in val:
                return

    col_names = [f"col{i}" for i in range(num_cols)]
    df = pd.DataFrame(data, columns=col_names)

    csv_str = df.to_csv(index=False)
    df_roundtrip = pd.read_csv(io.StringIO(csv_str))

    assert df.equals(df_roundtrip), f"Round-trip failed:\nOriginal:\n{df}\n\nRound-trip:\n{df_roundtrip}"
```

**Failing input**: `data=[['']]`

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame([['']],  columns=['col0'])
csv_str = df.to_csv(index=False)
df_roundtrip = pd.read_csv(io.StringIO(csv_str))

print("Original value:", repr(df.iloc[0, 0]))
print("After round-trip:", repr(df_roundtrip.iloc[0, 0]))
print("Round-trip preserves data:", df.equals(df_roundtrip))
```

Output:
```
Original value: ''
After round-trip: np.float64(nan)
Round-trip preserves data: False
```

## Why This Is A Bug

When users save a DataFrame to CSV and load it back, they expect to get the same data. Empty strings are valid string values and should be preserved, not converted to NaN. This violates the fundamental round-trip property that is essential for data persistence.

The CSV output correctly quotes empty strings as `""`, but `read_csv` with default settings interprets them as missing values. This behavior is not intuitive and can lead to silent data corruption in user workflows.

## Fix

The issue stems from the default `na_filter=True` parameter in `read_csv`, which treats empty fields as NA values. Users can work around this by using `na_filter=False` or `keep_default_na=False`, but this should be the default behavior when reading properly quoted empty strings.

A better default would be to only treat truly empty fields (not quoted) as NA, while preserving explicitly quoted empty strings. Alternatively, `to_csv` and `read_csv` should have symmetric defaults that preserve round-trip equality.

Workaround for users:
```python
df_roundtrip = pd.read_csv(io.StringIO(csv_str), na_filter=False)
```

This preserves empty strings correctly, though it also affects handling of other NA values like "NA", "None", etc.