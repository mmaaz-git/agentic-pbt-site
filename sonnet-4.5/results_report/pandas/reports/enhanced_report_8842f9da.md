# Bug Report: pandas.io.excel._util._excel2num Returns Negative Index for Empty String

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns `-1` for empty or whitespace-only strings instead of raising a `ValueError` as documented in its docstring, violating the function's contract and potentially causing negative indices to propagate through downstream code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.excel._util import _excel2num


@given(st.text())
def test_excel2num_never_returns_negative(col_name):
    """Test that _excel2num never returns negative values for any input."""
    try:
        result = _excel2num(col_name)
        assert result >= 0, (
            f"_excel2num('{col_name}') returned {result}, "
            "but column indices should never be negative"
        )
    except ValueError:
        pass  # ValueError is expected for invalid inputs
```

<details>

<summary>
**Failing input**: `""` (empty string)
</summary>
```
Running property-based test with Hypothesis...
Test failed: _excel2num('') returned -1, but column indices should never be negative
```
</details>

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num, _range2cols

# Test empty string
print("Testing _excel2num(''):")
try:
    result = _excel2num("")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test whitespace-only string
print("\nTesting _excel2num('   '):")
try:
    result = _excel2num("   ")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test tab character
print("\nTesting _excel2num('\\t'):")
try:
    result = _excel2num("\t")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test newline character
print("\nTesting _excel2num('\\n'):")
try:
    result = _excel2num("\n")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test downstream impact in _range2cols
print("\n\nDownstream impact in _range2cols:")
print(f"_range2cols('A,,C'): {_range2cols('A,,C')}")
print(f"_range2cols(',A'): {_range2cols(',A')}")
print(f"_range2cols('A,'): {_range2cols('A,')}")
print(f"_range2cols('   ,A'): {_range2cols('   ,A')}")

# Test with valid inputs for comparison
print("\n\nValid inputs for comparison:")
print(f"_excel2num('A'): {_excel2num('A')}")
print(f"_excel2num('B'): {_excel2num('B')}")
print(f"_excel2num('Z'): {_excel2num('Z')}")
print(f"_excel2num('AA'): {_excel2num('AA')}")
print(f"_excel2num('AB'): {_excel2num('AB')}")
```

<details>

<summary>
Empty strings return -1 instead of raising ValueError
</summary>
```
Testing _excel2num(''):
  Returned: -1

Testing _excel2num('   '):
  Returned: -1

Testing _excel2num('\t'):
  Returned: -1

Testing _excel2num('\n'):
  Returned: -1


Downstream impact in _range2cols:
_range2cols('A,,C'): [0, -1, 2]
_range2cols(',A'): [-1, 0]
_range2cols('A,'): [0, -1]
_range2cols('   ,A'): [-1, 0]


Valid inputs for comparison:
_excel2num('A'): 0
_excel2num('B'): 1
_excel2num('Z'): 25
_excel2num('AA'): 26
_excel2num('AB'): 27
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Contract Violation**: The function's docstring explicitly states it "Raises ValueError" when "Part of the Excel column name was invalid." An empty string has no valid parts and represents an invalid Excel column name, yet the function returns `-1` instead of raising the documented exception.

2. **Semantic Inconsistency**: Excel columns are labeled A, B, C, etc., starting from index 0. There is no Excel column that maps to index `-1`. The function is supposed to convert Excel column names to "0-based column index" per its documentation, but negative indices are outside this domain.

3. **Excel Specification Violation**: In Microsoft Excel, empty column names are invalid. When Excel encounters empty headers in spreadsheets, it automatically replaces them with placeholders like "Column1" or similar. The function should follow Excel's behavior by rejecting empty inputs.

4. **Downstream Data Corruption**: The `_range2cols` function, which depends on `_excel2num`, propagates these negative indices when parsing user input like `"A,,C"` (returning `[0, -1, 2]`). This can cause:
   - `IndexError` exceptions when the negative indices are used to access arrays
   - Silent data corruption if negative indexing wraps around to access wrong columns
   - Unexpected behavior in data processing pipelines

5. **Implementation Analysis**: The bug occurs because when `x.upper().strip()` produces an empty string, the for loop in lines 119-125 never executes, leaving `index = 0`. The function then returns `index - 1 = -1` on line 127, without any validation that the input was non-empty.

## Relevant Context

The `_excel2num` function is located in `/pandas/io/excel/_util.py` starting at line 98. While it's technically an internal function (prefixed with underscore), it's used by other pandas functions that are part of the public API, particularly for parsing Excel column specifications in the `usecols` parameter.

The function correctly handles invalid characters within non-empty strings (e.g., `_excel2num("A1")` raises `ValueError: Invalid column name: A1`), but fails to validate that the input contains at least one valid character.

Common scenarios where this bug could be triggered:
- User input with trailing commas: `usecols="A,B,"`
- Double commas from data entry errors: `usecols="A,,C"`
- Programmatically generated column lists with empty entries
- Whitespace-only entries from parsing formatted text

Documentation reference: The pandas.read_excel function accepts `usecols` parameter which can be a string of comma-separated Excel columns, making this bug reachable through normal pandas usage.

## Proposed Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -116,8 +116,12 @@ def _excel2num(x: str) -> int:
         Part of the Excel column name was invalid.
     """
     index = 0
+    x_stripped = x.upper().strip()
+
+    if not x_stripped:
+        raise ValueError(f"Invalid column name: {x!r}")

-    for c in x.upper().strip():
+    for c in x_stripped:
         cp = ord(c)

         if cp < ord("A") or cp > ord("Z"):
```