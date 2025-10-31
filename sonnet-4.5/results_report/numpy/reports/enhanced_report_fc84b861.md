# Bug Report: pandas.io.excel._util._range2cols Silently Returns Empty List for Reversed Column Ranges

**Target**: `pandas.io.excel._util._range2cols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_range2cols` function silently returns an empty list when given a reversed Excel column range (e.g., "C:A" instead of "A:C"), causing silent data loss without any warning or error.

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


if __name__ == "__main__":
    test_range2cols_handles_any_range_order()
```

<details>

<summary>
**Failing input**: `col1='A', col2='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 37, in <module>
    test_range2cols_handles_any_range_order()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 6, in test_range2cols_handles_any_range_order
    st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=3),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 33, in test_range2cols_handles_any_range_order
    assert len(result_reverse) == expected_length
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_range2cols_handles_any_range_order(
    # The test always failed when commented parts were varied together.
    col1='A',  # or any other generated value
    col2='B',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.excel._util import _range2cols

print("Testing _range2cols with forward and reversed ranges:")
print("=" * 60)

result_forward = _range2cols("A:C")
print(f"Forward range 'A:C' returns: {result_forward}")
print(f"  Expected: [0, 1, 2] (columns A, B, C)")
print(f"  Length: {len(result_forward)}")
print()

result_reversed = _range2cols("C:A")
print(f"Reversed range 'C:A' returns: {result_reversed}")
print(f"  Expected: [0, 1, 2] or [2, 1, 0] (columns C, B, A or A, B, C)")
print(f"  Length: {len(result_reversed)}")
print()

result_larger_reversed = _range2cols("AA:A")
print(f"Reversed range 'AA:A' returns: {result_larger_reversed}")
print(f"  Expected: list containing 27 elements (columns A through AA)")
print(f"  Length: {len(result_larger_reversed)}")
print()

result_complex = _range2cols("D:B,F,Z:AB")
print(f"Complex range 'D:B,F,Z:AB' returns: {result_complex}")
print(f"  Note: 'D:B' is reversed, should return columns B, C, D")
print(f"  Expected: B, C, D (1,2,3) + F (5) + Z, AA, AB (25,26,27)")
print(f"  Length: {len(result_complex)}")
```

<details>

<summary>
Silent failure - reversed ranges return empty lists
</summary>
```
Testing _range2cols with forward and reversed ranges:
============================================================
Forward range 'A:C' returns: [0, 1, 2]
  Expected: [0, 1, 2] (columns A, B, C)
  Length: 3

Reversed range 'C:A' returns: []
  Expected: [0, 1, 2] or [2, 1, 0] (columns C, B, A or A, B, C)
  Length: 0

Reversed range 'AA:A' returns: []
  Expected: list containing 27 elements (columns A through AA)
  Length: 0

Complex range 'D:B,F,Z:AB' returns: [5, 25, 26, 27]
  Note: 'D:B' is reversed, should return columns B, C, D
  Expected: B, C, D (1,2,3) + F (5) + Z, AA, AB (25,26,27)
  Length: 4
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Silent Data Loss**: The function returns an empty list without any warning, error, or exception when given a reversed range. Users expecting to read specific columns will silently receive no data, potentially leading to incorrect analysis or processing downstream.

2. **Inconsistent with Excel Behavior**: In Microsoft Excel, both "A:C" and "C:A" select the same columns (A, B, and C). Users familiar with Excel would reasonably expect pandas to handle ranges similarly when using `pd.read_excel()` with the `usecols` parameter.

3. **Documentation Mismatch**: The function's docstring shows examples like `_range2cols('A:E')` and `_range2cols('A,C,Z:AB')` but doesn't specify that ranges must be forward-ordered. The documentation implies the function converts "column ranges" to indices without restriction on ordering.

4. **Breaks Principle of Least Surprise**: When Python's built-in `range()` receives a start greater than stop, it returns an empty iterator, but that's documented behavior. Here, the function is specifically designed to handle Excel-style ranges where order shouldn't matter for selecting a set of columns.

5. **Affects Public API**: While `_range2cols` is technically private (underscore prefix), it's called by `maybe_convert_usecols` which is used by the public `pd.read_excel()` function. This means end users are affected when they specify column ranges like `usecols="C:A"`.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_util.py` at line 156. The problematic code uses Python's `range()` function directly:

```python
cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
```

When `_excel2num(rngs[0])` > `_excel2num(rngs[1])` (reversed range), `range()` returns an empty iterator as per Python's standard behavior.

The function is used in the data flow: `pd.read_excel()` → `maybe_convert_usecols()` → `_range2cols()`

Related pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

## Proposed Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -153,7 +153,11 @@ def _range2cols(areas: str) -> list[int]:
     for rng in areas.split(","):
         if ":" in rng:
             rngs = rng.split(":")
-            cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
+            start_idx = _excel2num(rngs[0])
+            end_idx = _excel2num(rngs[1])
+            # Normalize range order to handle reversed ranges like "C:A"
+            min_idx, max_idx = min(start_idx, end_idx), max(start_idx, end_idx)
+            cols.extend(range(min_idx, max_idx + 1))
         else:
             cols.append(_excel2num(rng))
```