# Bug Report: pandas.io.excel._util._excel2num Empty String Handling

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num()` function returns `-1` for empty strings instead of raising a `ValueError` as documented. This violates its contract and causes downstream issues in `_range2cols()`, which can affect the `read_excel()` function's `usecols` parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import string
from pandas.io.excel._util import _excel2num, _range2cols

@given(st.text(alphabet=string.ascii_uppercase + ',:', min_size=1, max_size=20))
def test_range2cols_sorted_and_unique(range_str):
    try:
        result = _range2cols(range_str)
        assert result == sorted(result), f"Result {result} is not sorted"
        assert len(result) == len(set(result)), f"Result {result} contains duplicates"
    except (ValueError, IndexError):
        pass
```

**Failing inputs**:
- `'A,'` produces `[0, -1]` (unsorted and invalid index)
- `','` produces `[-1, -1]` (duplicates and invalid index)

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num, _range2cols

result = _excel2num('')
print(f"_excel2num('') = {result}")

result = _range2cols('A,')
print(f"_range2cols('A,') = {result}")

result = _range2cols(',')
print(f"_range2cols(',') = {result}")
```

**Output**:
```
_excel2num('') = -1
_range2cols('A,') = [0, -1]
_range2cols(',') = [-1, -1]
```

## Why This Is A Bug

1. **Contract violation**: The `_excel2num()` docstring explicitly states it "Raises ValueError" for invalid column names, but empty strings are not rejected.

2. **Invalid output**: The function returns `-1` for empty strings, which is not a valid 0-based column index.

3. **Downstream impact**: This causes `_range2cols()` to produce invalid results when given trailing commas or empty elements, which can affect users of `read_excel()` with the `usecols` parameter.

4. **Silent failure**: The bug doesn't raise an error, making it harder to detect and debug.

## Fix

```diff
diff --git a/pandas/io/excel/_util.py b/pandas/io/excel/_util.py
index 1234567..abcdefg 100644
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -116,6 +116,10 @@ def _excel2num(x: str) -> int:
         Part of the Excel column name was invalid.
     """
     index = 0
+
+    # Check for empty string after stripping
+    if not x.strip():
+        raise ValueError(f"Invalid column name: {x}")

     for c in x.upper().strip():
         cp = ord(c)
```

Alternatively, `_range2cols()` could be modified to filter out empty strings before calling `_excel2num()`:

```diff
diff --git a/pandas/io/excel/_util.py b/pandas/io/excel/_util.py
index 1234567..abcdefg 100644
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -151,7 +151,10 @@ def _range2cols(areas: str) -> list[int]:
     cols: list[int] = []

     for rng in areas.split(","):
+        rng = rng.strip()
+        if not rng:
+            continue
         if ":" in rng:
             rngs = rng.split(":")
             cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
```