# Bug Report: pandas.io.excel Rows with All None Values Are Silently Dropped

**Target**: `pandas.DataFrame.to_excel()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a DataFrame to Excel using `to_excel()`, rows that consist entirely of `None` or `NaN` values are silently dropped and not written to the Excel file. This causes data loss and violates the fundamental expectation that the number of rows should be preserved during serialization.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes
import pandas as pd
import tempfile
import os


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(
        st.one_of(
            st.integers(min_value=-1000, max_value=1000),
            st.none()
        ),
        min_size=1,
        max_size=20
    )
)
def test_excel_none_values_roundtrip(values):
    df = pd.DataFrame({'col': values})

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, index=False)
        result = pd.read_excel(tmp_path)

        assert len(df) == len(result), f"Expected {len(df)} rows, got {len(result)}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `values=[None]` (or any list with only None values)

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import openpyxl

df = pd.DataFrame({'col': [None, None, None]})
print(f"Original DataFrame: {len(df)} rows")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

df.to_excel(tmp_path, index=False)

wb = openpyxl.load_workbook(tmp_path)
ws = wb.active
print(f"Excel file: {ws.max_row - 1} data rows (max_row={ws.max_row} includes header)")

result = pd.read_excel(tmp_path)
print(f"Result DataFrame: {len(result)} rows")
```

Output:
```
Original DataFrame: 3 rows
Excel file: 0 data rows (max_row=1 includes header)
Result DataFrame: 0 rows
```

Additional examples demonstrating the pattern:
- `[None]` → 0 rows written
- `[None, None, None]` → 0 rows written
- `[1, None]` → 1 row written (trailing None row dropped)
- `[None, 1]` → 2 rows written (both preserved)
- Two columns `[None], [None]` → 0 rows written

## Why This Is A Bug

1. **Data Loss**: Rows are silently dropped without warning or error
2. **Violates Round-Trip Property**: `read_excel(to_excel(df))` returns a different number of rows than the original DataFrame
3. **No Workaround**: Parameters like `na_filter=False` and `keep_default_na=False` do not prevent this behavior
4. **Breaking Invariant**: The fundamental invariant `len(df) == len(read_excel(to_excel(df)))` is violated
5. **Silent Failure**: Users have no indication that data has been lost

This is particularly problematic because:
- In data pipelines, row counts are often used for validation
- Rows with missing data are semantically different from no rows at all
- The row index/position carries meaning that is lost when rows are dropped

## Fix

The root cause is that `to_excel()` doesn't write rows when all cells contain None/NaN values. The fix should ensure that all rows are written, even if they contain only None/NaN values.

A high-level fix would be to modify the Excel writer to:
1. Always write the correct number of rows, regardless of whether cells contain None/NaN
2. Write None/NaN values as empty cells in Excel (which is already the expected behavior)
3. Ensure that trailing rows with all-None values are not skipped

This likely requires changes in the `pandas/io/excel/_base.py` or engine-specific writers (e.g., `_xlsxwriter.py`, `_openpyxl.py`) to ensure they iterate over all rows in the DataFrame, not just rows with non-None values.