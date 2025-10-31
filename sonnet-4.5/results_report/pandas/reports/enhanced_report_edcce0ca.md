# Bug Report: pandas.io.excel._util.fill_mi_header Fails to Forward Fill None Values When control_row is False

**Target**: `pandas.io.excel._util.fill_mi_header`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fill_mi_header` function fails to forward-fill None values when the corresponding `control_row[i]` value is False, violating its documented purpose of "forward filling blank entries in row."

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
        assert val is not None, f"Position {i} should not be None after forward fill. Input: row={row_with_blanks}, control_row={control_row}"
```

<details>

<summary>
**Failing input**: `row=[1, 1], control_row=[False, False]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 22, in <module>
    test_fill_mi_header_forward_fill_semantics()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 5, in test_fill_mi_header_forward_fill_semantics
    st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 18, in test_fill_mi_header_forward_fill_semantics
    assert val is not None, f"Position {i} should not be None after forward fill. Input: row={row_with_blanks}, control_row={control_row}"
           ^^^^^^^^^^^^^^^
AssertionError: Position 1 should not be None after forward fill. Input: row=[1, None], control_row=[False, False]
Falsifying example: test_fill_mi_header_forward_fill_semantics(
    row=[1, 1],
    control_row=[False, False],
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.excel._util import fill_mi_header

# Test case demonstrating the bug
row = [1, None]
control_row = [False, False]

print(f"Input row: {row}")
print(f"Input control_row: {control_row}")

result_row, result_control = fill_mi_header(row.copy(), control_row.copy())

print(f"\nOutput row: {result_row}")
print(f"Output control_row: {result_control}")

print(f"\nExpected row: [1, 1] (forward fill None with 1)")
print(f"Actual row: {result_row}")

# Check if the forward fill worked correctly
if result_row[1] == 1:
    print("\n✓ Forward fill worked correctly")
else:
    print(f"\n✗ Forward fill failed: Expected row[1]=1, but got row[1]={result_row[1]}")

# Additional test case with multiple Nones
print("\n" + "="*50)
print("Additional test case with multiple None values:")

row2 = [5, None, None, 10, None]
control_row2 = [False, False, False, False, False]

print(f"\nInput row: {row2}")
print(f"Input control_row: {control_row2}")

result_row2, result_control2 = fill_mi_header(row2.copy(), control_row2.copy())

print(f"\nOutput row: {result_row2}")
print(f"Output control_row: {result_control2}")

print(f"\nExpected row: [5, 5, 5, 10, 10] (forward fill Nones)")
print(f"Actual row: {result_row2}")

# Check if all Nones were filled
expected2 = [5, 5, 5, 10, 10]
if result_row2 == expected2:
    print("\n✓ All forward fills worked correctly")
else:
    print(f"\n✗ Forward fill failed: Expected {expected2}, but got {result_row2}")
```

<details>

<summary>
Forward fill fails when control_row is False and value is None
</summary>
```
Input row: [1, None]
Input control_row: [False, False]

Output row: [1, None]
Output control_row: [False, False]

Expected row: [1, 1] (forward fill None with 1)
Actual row: [1, None]

✗ Forward fill failed: Expected row[1]=1, but got row[1]=None

==================================================
Additional test case with multiple None values:

Input row: [5, None, None, 10, None]
Input control_row: [False, False, False, False, False]

Output row: [5, None, None, 10, None]
Output control_row: [False, False, False, False, False]

Expected row: [5, 5, 5, 10, 10] (forward fill Nones)
Actual row: [5, None, None, 10, None]

✗ Forward fill failed: Expected [5, 5, 5, 10, 10], but got [5, None, None, 10, None]
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it should "Forward fill blank entries in row but only inside the same parent index." However, the current implementation contains a critical logic flaw at lines 264-265 of `/pandas/io/excel/_util.py`:

```python
if not control_row[i]:
    last = row[i]  # BUG: Sets last=None when row[i] is None
```

When `control_row[i]` is False (indicating a potential parent index boundary) and `row[i]` is None, the code incorrectly updates `last = None`. This breaks the forward-fill chain because the subsequent check `if row[i] is None: row[i] = last` attempts to fill None with None, leaving the blank entry unfilled.

The function violates its documented contract in the following ways:
1. **Blank entries remain blank**: None values are not filled when they coincide with False control_row values
2. **Forward-fill chain is broken**: The "last valid value" is incorrectly reset to None instead of being preserved
3. **Inconsistent behavior**: The function only works correctly when control_row values are True or when non-None values appear at False positions

## Relevant Context

This function is critical for parsing Excel files with MultiIndex headers, particularly when dealing with merged cells that create blank entries. The bug can lead to:
- Incorrect data structure when reading Excel files with complex headers
- Missing values in MultiIndex column headers that should have been filled
- Downstream errors in data processing pipelines that expect complete header information

The function is located in `/pandas/io/excel/_util.py` (lines 241-273) and is used by the Excel parsing infrastructure in pandas, specifically in `/pandas/io/excel/_base.py`.

Documentation reference: The function is an internal utility but affects the public API behavior of `pandas.read_excel()` when parsing files with MultiIndex headers.

## Proposed Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -261,7 +261,8 @@ def fill_mi_header(
     """
     last = row[0]
     for i in range(1, len(row)):
-        if not control_row[i]:
+        # Only update 'last' at boundaries when we have a non-blank value
+        if not control_row[i] and row[i] is not None and row[i] != "":
             last = row[i]

         if row[i] == "" or row[i] is None:
```