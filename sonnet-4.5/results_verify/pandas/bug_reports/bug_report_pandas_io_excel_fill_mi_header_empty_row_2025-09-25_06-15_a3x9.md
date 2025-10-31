# Bug Report: pandas.io.excel._util.fill_mi_header Empty Row Crash

**Target**: `pandas.io.excel._util.fill_mi_header`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `fill_mi_header` function crashes with an `IndexError` when given an empty row, instead of handling it gracefully or providing a clear error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.excel._util import fill_mi_header


@settings(max_examples=1000)
@given(st.lists(st.one_of(st.text(max_size=5), st.just(""), st.none()), min_size=0, max_size=20))
def test_fill_mi_header_handles_all_input_sizes(row):
    """
    Property: fill_mi_header should either handle all row sizes gracefully
    or raise an informative error (not IndexError)
    """
    control_row = [True] * len(row)

    try:
        result_row, result_control = fill_mi_header(row.copy(), control_row.copy())
        assert len(result_row) == len(row)
        assert len(result_control) == len(control_row)
    except IndexError:
        raise AssertionError(f"fill_mi_header should not raise IndexError for row of length {len(row)}")
```

**Failing input**: `row = []`, `control_row = []`

## Reproducing the Bug

```python
from pandas.io.excel._util import fill_mi_header

row = []
control_row = []
result_row, result_control = fill_mi_header(row, control_row)
```

Output:
```
Traceback (most recent call last):
  ...
  File "pandas/io/excel/_util.py", line 262, in fill_mi_header
    last = row[0]
IndexError: list index out of range
```

Expected: Either handle empty rows gracefully by returning `([], [])`, or raise a `ValueError` with a clear message like "Cannot fill header for empty row"

## Why This Is A Bug

The function unconditionally accesses `row[0]` without checking if the row is non-empty. This violates defensive programming principles:

1. **Poor error handling**: `IndexError` is cryptic - it doesn't tell the user what went wrong
2. **Inconsistent with Python conventions**: Empty sequences are typically handled as edge cases, not errors
3. **Unexpected crash**: The function signature doesn't indicate that empty rows are invalid

Looking at the implementation:

```python
def fill_mi_header(
    row: list[Hashable], control_row: list[bool]
) -> tuple[list[Hashable], list[bool]]:
    last = row[0]  # ← Crashes here if row is empty
    for i in range(1, len(row)):
        ...
    return row, control_row
```

The function is used for forward-filling blank entries in MultiIndex headers. For an empty header row, the logical behavior would be to return it unchanged.

## Severity Justification

Rated as **Low** severity because:
- This is likely an internal utility function
- Empty header rows are probably rare in real Excel files
- Callers may already validate that rows are non-empty

However, it's still worth fixing for robustness and better error messages.

## Fix

```diff
 def fill_mi_header(
     row: list[Hashable], control_row: list[bool]
 ) -> tuple[list[Hashable], list[bool]]:
+    if not row:
+        return row, control_row
+
     last = row[0]
     for i in range(1, len(row)):
         if not control_row[i]:
             last = row[i]

         if row[i] == "" or row[i] is None:
             row[i] = last
         else:
             control_row[i] = False
             last = row[i]

     return row, control_row
```