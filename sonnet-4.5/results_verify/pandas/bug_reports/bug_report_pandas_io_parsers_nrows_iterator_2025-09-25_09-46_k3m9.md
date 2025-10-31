# Bug Report: pandas.io.parsers nrows Parameter Ignored with iterator=True

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `nrows` parameter is ignored when `iterator=True` is passed to `read_csv` and the `.read()` method is called on the resulting `TextFileReader`. Instead of limiting the number of rows read, it reads all available data.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import io

@settings(max_examples=200)
@given(
    nrows=st.integers(min_value=1, max_value=20),
    total_rows=st.integers(min_value=1, max_value=50)
)
def test_iterator_nrows_respected(nrows, total_rows):
    csv_lines = ["col1,col2"]
    for i in range(total_rows):
        csv_lines.append(f"{i},{i*2}")
    csv_data = "\n".join(csv_lines)

    reader = pd.read_csv(io.StringIO(csv_data), iterator=True, nrows=nrows)
    result = reader.read()

    expected_rows = min(nrows, total_rows)
    assert len(result) == expected_rows, (
        f"nrows parameter should be respected with iterator=True. "
        f"Expected {expected_rows} rows, got {len(result)}"
    )
```

**Failing input**: `nrows=1, total_rows=2`

## Reproducing the Bug

```python
import pandas as pd
import io

csv_data = "a,b\n1,2\n3,4\n5,6\n"

print("Expected behavior: nrows=1 should read only 1 row")
df_normal = pd.read_csv(io.StringIO(csv_data), nrows=1)
print(f"read_csv(nrows=1): {len(df_normal)} rows")

print("\nActual behavior: nrows parameter ignored with iterator=True")
reader = pd.read_csv(io.StringIO(csv_data), iterator=True, nrows=1)
df_iterator = reader.read()
print(f"read_csv(iterator=True, nrows=1).read(): {len(df_iterator)} rows")

print(f"\nBUG: Expected 1 row, got {len(df_iterator)} rows")
```

## Why This Is A Bug

The `nrows` parameter is documented to limit the number of rows to read from the CSV file. This works correctly in all other contexts:
- `read_csv(nrows=N)` returns N rows ✓
- `read_csv(chunksize=C, nrows=N)` reads N rows total across chunks ✓
- `read_csv(iterator=True, nrows=N).read()` reads ALL rows ✗

This is a clear contract violation. The parameter is accepted but ignored, violating the documented behavior and user expectations.

## Fix

The issue is in the `TextFileReader` class. When `iterator=True` is specified (without `chunksize`), the `nrows` parameter should still be respected by the `.read()` method. The `nrows` value is likely being stored but not used when `.read()` is called on an iterator.

The fix would involve ensuring that `TextFileReader.read()` respects the `nrows` limit that was set during initialization. This likely requires checking `self.nrows` in the `read()` method implementation and limiting the rows read accordingly.