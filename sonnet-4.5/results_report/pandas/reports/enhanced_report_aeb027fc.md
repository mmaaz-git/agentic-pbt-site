# Bug Report: pandas.io.excel._util._excel2num Returns Invalid Index for Empty Strings

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns -1 for empty or whitespace-only strings instead of raising a `ValueError`, causing invalid column indices to propagate through Excel parsing operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from pandas.io.excel._util import _excel2num

@given(st.text())
@example('')  # Force testing with empty string
@example('   ')  # Force testing with whitespace
@settings(max_examples=100)
def test_excel2num_valid_or_error(col_name):
    """Test that _excel2num either returns a valid index or raises ValueError"""
    try:
        result = _excel2num(col_name)
        assert result >= 0, f"Column index must be non-negative, got {result} for input '{col_name}'"
    except ValueError:
        # ValueError is acceptable for invalid inputs
        pass

if __name__ == "__main__":
    test_excel2num_valid_or_error()
```

<details>

<summary>
**Failing input**: `''` (empty string) and `'   '` (whitespace-only string)
</summary>
```
+ Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 18, in <module>
  |     test_excel2num_valid_or_error()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 5, in test_excel2num_valid_or_error
  |     @example('')  # Force testing with empty string
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures in explicit examples. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 12, in test_excel2num_valid_or_error
    |     assert result >= 0, f"Column index must be non-negative, got {result} for input '{col_name}'"
    |            ^^^^^^^^^^^
    | AssertionError: Column index must be non-negative, got -1 for input ''
    | Falsifying explicit example: test_excel2num_valid_or_error(
    |     col_name='',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 12, in test_excel2num_valid_or_error
    |     assert result >= 0, f"Column index must be non-negative, got {result} for input '{col_name}'"
    |            ^^^^^^^^^^^
    | AssertionError: Column index must be non-negative, got -1 for input '   '
    | Falsifying explicit example: test_excel2num_valid_or_error(
    |     col_name='   ',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num

# Test with empty string
print("Testing _excel2num with empty string:")
result1 = _excel2num('')
print(f"_excel2num('') = {result1}")

# Test with whitespace-only string
print("\nTesting _excel2num with whitespace-only string:")
result2 = _excel2num('   ')
print(f"_excel2num('   ') = {result2}")

# Test with valid column names for comparison
print("\nTesting _excel2num with valid column names:")
result3 = _excel2num('A')
print(f"_excel2num('A') = {result3}")

result4 = _excel2num('B')
print(f"_excel2num('B') = {result4}")

result5 = _excel2num('Z')
print(f"_excel2num('Z') = {result5}")

result6 = _excel2num('AA')
print(f"_excel2num('AA') = {result6}")

# Test with tab character
print("\nTesting _excel2num with tab character:")
result7 = _excel2num('\t')
print(f"_excel2num('\\t') = {result7}")

# Test how this affects _range2cols
from pandas.io.excel._util import _range2cols

print("\nTesting how empty strings affect _range2cols:")
result8 = _range2cols(',')  # This should have an empty string between commas
print(f"_range2cols(',') = {result8}")

result9 = _range2cols('A,,B')  # Empty column between A and B
print(f"_range2cols('A,,B') = {result9}")
```

<details>

<summary>
Output demonstrating invalid -1 indices returned
</summary>
```
Testing _excel2num with empty string:
_excel2num('') = -1

Testing _excel2num with whitespace-only string:
_excel2num('   ') = -1

Testing _excel2num with valid column names:
_excel2num('A') = 0
_excel2num('B') = 1
_excel2num('Z') = 25
_excel2num('AA') = 26

Testing _excel2num with tab character:
_excel2num('\t') = -1

Testing how empty strings affect _range2cols:
_range2cols(',') = [-1, -1]
_range2cols('A,,B') = [0, -1, 1]
```
</details>

## Why This Is A Bug

The `_excel2num` function is documented to "Convert Excel column name like 'AB' to 0-based column index" and should raise a `ValueError` when "Part of the Excel column name was invalid." However, for empty or whitespace-only strings, it returns -1 instead of raising an error. This violates several expectations:

1. **Invalid Return Value**: The function promises to return "The column index corresponding to the name" (lines 109-110 in _util.py). A -1 is not a valid 0-based column index in Excel.

2. **Inconsistent Error Handling**: The function correctly validates that characters must be A-Z and raises `ValueError` for invalid characters (lines 122-123), but fails to validate that the input is non-empty after stripping whitespace.

3. **Silent Propagation**: When used by `_range2cols` (lines 156, 158), invalid -1 indices are silently added to column lists, which could cause:
   - Incorrect column selection when using Python's negative indexing (where -1 means last column)
   - Index errors in downstream code expecting valid non-negative indices
   - Data corruption if these indices are used for Excel file operations

4. **Algorithm Logic Flaw**: The function iterates over `x.upper().strip()` (line 119). When this results in an empty string, the loop never executes, leaving `index` at its initial value of 0. The function then returns `index - 1` (line 127), yielding -1.

## Relevant Context

The function is located at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_util.py:98-127`

This is an internal utility function (indicated by the underscore prefix) used by pandas for Excel file operations. The `_range2cols` function (lines 130-160) directly uses `_excel2num` to parse column specifications like "A:E" or "A,C,Z:AB". When empty strings appear in column specifications (e.g., "A,,B"), they result in -1 values in the column index list.

Documentation for the pandas Excel functionality: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

## Proposed Fix

```diff
def _excel2num(x: str) -> int:
    """
    Convert Excel column name like 'AB' to 0-based column index.

    Parameters
    ----------
    x : str
        The Excel column name to convert to a 0-based column index.

    Returns
    -------
    num : int
        The column index corresponding to the name.

    Raises
    ------
    ValueError
        Part of the Excel column name was invalid.
    """
+   x = x.upper().strip()
+
+   if not x:
+       raise ValueError(f"Invalid column name: empty string")
+
    index = 0

-   for c in x.upper().strip():
+   for c in x:
        cp = ord(c)

        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")

        index = index * 26 + cp - ord("A") + 1

    return index - 1
```