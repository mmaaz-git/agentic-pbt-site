# Bug Report: pandas.io.excel._util._range2cols Reverse Range Handling

**Target**: `pandas.io.excel._util._range2cols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_range2cols` function silently returns an empty list when given a reverse range (e.g., "C:A" instead of "A:C"), rather than either handling it correctly or raising a clear error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, example
import string
from pandas.io.excel._util import _range2cols, _excel2num

@given(
    start_col=st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=2),
    end_col=st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=2)
)
@example(start_col='C', end_col='A')
def test_range2cols_reverse_range(start_col, end_col):
    try:
        start_idx = _excel2num(start_col)
        end_idx = _excel2num(end_col)
    except ValueError:
        assume(False)

    if start_idx > end_idx:
        result = _range2cols(f"{start_col}:{end_col}")
        assert len(result) > 0, f"Reverse range returned empty list"
```

**Failing input**: `"C:A"`

## Reproducing the Bug

```python
from pandas.io.excel._util import _range2cols

result = _range2cols('C:A')
print(result)

assert result == [], "Returns empty list"
assert len(result) == 0, "Should contain 3 elements [2, 1, 0] or [0, 1, 2]"
```

## Why This Is A Bug

The function should either:
1. Interpret reverse ranges as reversed (i.e., "C:A" → [0, 1, 2] or [2, 1, 0])
2. Raise a clear `ValueError` indicating that the range is invalid

Instead, it silently returns an empty list because the underlying Python `range()` function returns an empty range when `start > stop`. This can lead to confusing behavior where columns are silently ignored.

Looking at the code in `_range2cols` (lines 154-156):

```python
if ":" in rng:
    rngs = rng.split(":")
    cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
```

When `rngs[0]` = "C" (index 2) and `rngs[1]` = "A" (index 0), this becomes `range(2, 1)` which is empty.

The docstring examples only show forward ranges:
- `_range2cols('A:E')` → `[0, 1, 2, 3, 4]`
- `_range2cols('A,C,Z:AB')` → `[0, 2, 25, 26, 27]`

There's no documentation of how reverse ranges should behave, but returning an empty list is clearly wrong.

## Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -153,7 +153,12 @@ def _range2cols(areas: str) -> list[int]:
     for rng in areas.split(","):
         if ":" in rng:
             rngs = rng.split(":")
-            cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
+            start = _excel2num(rngs[0])
+            end = _excel2num(rngs[1])
+            if start > end:
+                raise ValueError(
+                    f"Invalid column range '{rng}': start column '{rngs[0]}' comes after end column '{rngs[1]}'"
+                )
+            cols.extend(range(start, end + 1))
         else:
             cols.append(_excel2num(rng))