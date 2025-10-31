# Bug Report: pandas.io.excel._util._range2cols Reversed Range Bug

**Target**: `pandas.io.excel._util._range2cols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_range2cols` function silently returns an empty list when given a reversed Excel column range (e.g., "C:A" instead of "A:C"), instead of either handling the range correctly or raising a clear error. This leads to unexpected behavior where users might think they're selecting columns but get nothing.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from pandas.io.excel._util import _range2cols, _excel2num


@given(
    st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=3),
    st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=3)
)
@settings(max_examples=1000)
def test_range2cols_handles_any_range_order(col1, col2):
    """
    Property: _range2cols should handle ranges in any order.
    Either both forward and reversed ranges should work,
    or reversed ranges should raise an error.
    Silently returning an empty list is a bug.
    """
    idx1 = _excel2num(col1)
    idx2 = _excel2num(col2)

    assume(idx1 != idx2)

    forward_range = f"{col1}:{col2}" if idx1 < idx2 else f"{col2}:{col1}"
    reverse_range = f"{col2}:{col1}" if idx1 < idx2 else f"{col1}:{col2}"

    result_forward = _range2cols(forward_range)
    result_reverse = _range2cols(reverse_range)

    min_idx = min(idx1, idx2)
    max_idx = max(idx1, idx2)
    expected_length = max_idx - min_idx + 1

    assert len(result_forward) == expected_length
    assert len(result_reverse) == expected_length
```

**Failing input**: `col1='A', col2='B'` (which creates reversed range "B:A")

## Reproducing the Bug

```python
from pandas.io.excel._util import _range2cols

result_forward = _range2cols("A:C")
print(f"Forward range A:C = {result_forward}")

result_reversed = _range2cols("C:A")
print(f"Reversed range C:A = {result_reversed}")

result_larger = _range2cols("AA:A")
print(f"Reversed range AA:A = {result_larger}")
```

**Output:**
```
Forward range A:C = [0, 1, 2]
Reversed range C:A = []
Reversed range AA:A = []
```

## Why This Is A Bug

1. **Excel compatibility**: In Microsoft Excel, both "A:C" and "C:A" select the same columns. Users expect pandas to behave similarly.

2. **Silent failure**: The function returns an empty list without any warning or error, making it difficult to debug when users accidentally specify a reversed range.

3. **Inconsistent behavior**: The function successfully handles forward ranges but silently fails on reversed ranges, leading to confusing and unpredictable behavior.

4. **Impact on `read_excel`**: This affects the `usecols` parameter in `pd.read_excel()`, where users specify column ranges as strings. A reversed range will silently select no columns instead of the intended columns.

## Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -151,7 +151,13 @@ def _range2cols(areas: str) -> list[int]:

     for rng in areas.split(","):
         if ":" in rng:
             rngs = rng.split(":")
-            cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
+            start_idx = _excel2num(rngs[0])
+            end_idx = _excel2num(rngs[1])
+
+            if start_idx > end_idx:
+                start_idx, end_idx = end_idx, start_idx
+
+            cols.extend(range(start_idx, end_idx + 1))
         else:
             cols.append(_excel2num(rng))

```