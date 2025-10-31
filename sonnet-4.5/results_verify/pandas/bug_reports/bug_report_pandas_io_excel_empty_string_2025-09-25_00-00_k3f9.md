# Bug Report: pandas.io.excel Empty String Data Loss

**Target**: `pandas.io.excel.to_excel` and `pandas.io.excel.read_excel`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Empty strings in DataFrames are silently converted to NaN during Excel round-trip operations, violating the fundamental property that `read_excel(to_excel(df))` should preserve the original data.

## Property-Based Test

```python
import tempfile
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@settings(max_examples=100, deadline=None)
@given(
    data=st.lists(
        st.lists(
            st.one_of(
                st.integers(min_value=-1e6, max_value=1e6),
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=20),
                st.booleans(),
            ),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=20,
    )
)
def test_round_trip_basic(data):
    if not all(len(row) == len(data[0]) for row in data):
        return

    df_original = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_original.to_excel(tmp_path, index=False)
        df_read = pd.read_excel(tmp_path)

        assert_frame_equal(df_original, df_read, check_dtype=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `data=[['']]`

## Reproducing the Bug

```python
import tempfile
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'status': ['active', '']
})

print("Original DataFrame:")
print(df)

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    filepath = f.name

df.to_excel(filepath, index=False)
df_read = pd.read_excel(filepath)

print("\nAfter round-trip:")
print(df_read)

print("\nData loss:")
print(f"Original: {df['status'].tolist()}")
print(f"After: {df_read['status'].tolist()}")
```

Output:
```
Original DataFrame:
    name  status
0  Alice  active
1    Bob

After round-trip:
    name  status
0  Alice  active
1    Bob     NaN

Data loss:
Original: ['active', '']
After: ['active', nan]
```

## Why This Is A Bug

This violates the fundamental round-trip property that users expect from serialization operations. When a user writes a DataFrame to Excel and reads it back, they expect to get the same data. Empty strings are valid data that should be preserved, not silently converted to NaN.

The bug occurs in two stages:
1. `to_excel()` writes empty strings as empty cells (None in Excel)
2. `read_excel()` interprets empty cells as NaN by default

Even with `keep_default_na=False` and `na_filter=False`, rows containing only empty strings may be completely lost, as they are not written to the Excel file at all.

This is particularly problematic because:
- Empty strings and NaN have different semantics (explicit empty vs missing data)
- The transformation is silent with no warnings
- There is no obvious workaround for users
- Data integrity is compromised

## Fix

The fix should be in `read_excel()` to distinguish between empty cells that were originally empty strings vs cells that represent true missing values. One approach:

Add a parameter like `empty_as_string=True` that treats empty cells as empty strings rather than NaN. Or better, make this the default behavior to match user expectations.

Alternatively, `to_excel()` could write a special marker for empty strings that `read_excel()` recognizes, though this would require changes to both functions.