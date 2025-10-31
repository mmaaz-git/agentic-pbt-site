# Bug Report: pandas.io.excel._util._range2cols Duplicate Columns

**Target**: `pandas.io.excel._util._range2cols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_range2cols` function returns duplicate column indices when given overlapping ranges or repeated column specifications. This causes the Excel reader to include the same columns multiple times in the output DataFrame.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
from pandas.io.excel._util import _range2cols

def num2excel(n):
    result = ''
    n = n + 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result

@given(st.integers(min_value=0, max_value=50), st.integers(min_value=1, max_value=50))
def test_range2cols_no_duplicates(start, overlap_size):
    end1 = start + overlap_size
    start2 = start + (overlap_size // 2)
    end2 = start2 + overlap_size
    range_str = f"{num2excel(start)}:{num2excel(end1)},{num2excel(start2)}:{num2excel(end2)}"
    result = _range2cols(range_str)
    assert len(result) == len(set(result)), \
        f"Overlapping ranges should not create duplicates: {range_str} -> {result}"
```

**Failing input**: `'A:C,B:D'` returns `[0, 1, 2, 1, 2, 3]`

## Reproducing the Bug

```python
from pandas.io.excel._util import _range2cols

result1 = _range2cols('A:C,B:D')
print(f"_range2cols('A:C,B:D') = {result1}")
print(f"Has duplicates: {len(result1) != len(set(result1))}")

result2 = _range2cols('A,A,A')
print(f"_range2cols('A,A,A') = {result2}")

result3 = _range2cols('A,B,C,B:D')
print(f"_range2cols('A,B,C,B:D') = {result3}")
```

**Output:**
```
_range2cols('A:C,B:D') = [0, 1, 2, 1, 2, 3]
Has duplicates: True
_range2cols('A,A,A') = [0, 0, 0]
_range2cols('A,B,C,B:D') = [0, 1, 2, 1, 2, 3]
```

## Why This Is A Bug

The documentation for `read_excel`'s `usecols` parameter states it "Returns a subset of the columns", implying each column should appear at most once. Users expect column specifications to behave like sets (no duplicates), similar to how other pandas APIs handle column selection.

When duplicates are passed to the Excel parser, it may:
1. Read the same column data multiple times
2. Create duplicate column names in the resulting DataFrame
3. Waste processing time and memory

This violates the principle of least surprise - users specifying overlapping ranges like `'A:E,C:G'` expect to get columns A through G once, not with C-E appearing twice.

## Fix

```diff
def _range2cols(areas: str) -> list[int]:
-   cols: list[int] = []
+   cols: set[int] = set()

    for rng in areas.split(","):
        if ":" in rng:
            rngs = rng.split(":")
-           cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
+           cols.update(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
        else:
-           cols.append(_excel2num(rng))
+           cols.add(_excel2num(rng))

-   return cols
+   return sorted(cols)
```

Note: Sorting ensures consistent ordering, matching the behavior shown in the docstring example.