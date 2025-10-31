# Bug Report: pandas.io.excel._util.fill_mi_header Forward Fill Fails at Boundaries

**Target**: `pandas.io.excel._util.fill_mi_header`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fill_mi_header` function fails to forward-fill None values when `control_row[i]` is False and `row[i]` is None. Instead of preserving the last valid value for forward filling, it resets `last` to None, breaking the forward-fill chain.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
from pandas.io.excel._util import fill_mi_header

@given(
    st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=20),
    st.lists(st.booleans(), min_size=2, max_size=20)
)
def test_fill_mi_header_forward_fill_semantics(row, control_row):
    assume(len(row) == len(control_row))

    row_with_blanks = row.copy()
    for i in range(1, len(row_with_blanks), 3):
        row_with_blanks[i] = None

    result_row, _ = fill_mi_header(row_with_blanks, control_row.copy())

    for i, val in enumerate(result_row):
        assert val is not None
```

**Failing input**: `row=[1, 1], control_row=[False, False]` (after setting row[1] to None)

## Reproducing the Bug

```python
from pandas.io.excel._util import fill_mi_header

row = [1, None]
control_row = [False, False]

result_row, result_control = fill_mi_header(row, control_row)

print(f"Input: {[1, None]}")
print(f"Output: {result_row}")
print(f"Expected: [1, 1] (forward fill None with 1)")
print(f"Actual: {result_row}")

assert result_row[1] == 1, f"Expected forward fill, but got {result_row[1]}"
```

## Why This Is A Bug

The function's docstring states it should "Forward fill blank entries in row but only inside the same parent index." However, when `control_row[i]` is False (indicating a boundary from a previous call) and `row[i]` is None, the code incorrectly sets `last = None`, which prevents forward filling for the None value.

The problematic code sequence:
1. `if not control_row[i]: last = row[i]` - Sets `last = None` when row[i] is None
2. `if row[i] is None: row[i] = last` - Tries to fill with None, which doesn't help

This violates the expectation that blank entries should be forward-filled with the last non-blank value.

## Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -261,7 +261,7 @@ def fill_mi_header(
     """
     last = row[0]
     for i in range(1, len(row)):
-        if not control_row[i]:
+        if not control_row[i] and row[i] is not None and row[i] != "":
             last = row[i]

         if row[i] == "" or row[i] is None:
```

The fix ensures that `last` is only updated when we encounter a boundary with an actual (non-blank) value. This preserves the last valid value for forward-filling subsequent None values.