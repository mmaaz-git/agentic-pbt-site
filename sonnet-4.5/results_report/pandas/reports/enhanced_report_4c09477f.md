# Bug Report: pandas.io.excel._util._excel2num Empty String Contract Violation

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_excel2num()` function returns `-1` for empty or whitespace-only strings instead of raising `ValueError` as documented, causing invalid column indices to propagate through `_range2cols()`.

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

# Run the test
if __name__ == "__main__":
    test_range2cols_sorted_and_unique()
```

<details>

<summary>
**Failing input**: `'A,'` and `','`
</summary>
```
+ Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 16, in <module>
  |     test_range2cols_sorted_and_unique()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 6, in test_range2cols_sorted_and_unique
  |     def test_range2cols_sorted_and_unique(range_str):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 9, in test_range2cols_sorted_and_unique
    |     assert result == sorted(result), f"Result {result} is not sorted"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Result [0, -1] is not sorted
    | Falsifying example: test_range2cols_sorted_and_unique(
    |     range_str='A,',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 10, in test_range2cols_sorted_and_unique
    |     assert len(result) == len(set(result)), f"Result {result} contains duplicates"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Result [-1, -1] contains duplicates
    | Falsifying example: test_range2cols_sorted_and_unique(
    |     range_str=',',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num, _range2cols

# Test _excel2num with empty string
print("Testing _excel2num with empty string:")
try:
    result = _excel2num('')
    print(f"_excel2num('') = {result}")
except ValueError as e:
    print(f"_excel2num('') raised ValueError: {e}")

# Test _excel2num with whitespace-only string
print("\nTesting _excel2num with whitespace-only string:")
try:
    result = _excel2num('   ')
    print(f"_excel2num('   ') = {result}")
except ValueError as e:
    print(f"_excel2num('   ') raised ValueError: {e}")

# Test _range2cols with trailing comma
print("\nTesting _range2cols with trailing comma:")
result = _range2cols('A,')
print(f"_range2cols('A,') = {result}")

# Test _range2cols with only comma
print("\nTesting _range2cols with only comma:")
result = _range2cols(',')
print(f"_range2cols(',') = {result}")

# Test _range2cols with multiple trailing commas
print("\nTesting _range2cols with multiple trailing commas:")
result = _range2cols('A,,')
print(f"_range2cols('A,,') = {result}")

# Test _range2cols with comma between valid columns
print("\nTesting _range2cols with empty element between valid columns:")
result = _range2cols('A,,B')
print(f"_range2cols('A,,B') = {result}")

# Test that valid inputs still work
print("\nTesting valid inputs:")
print(f"_excel2num('A') = {_excel2num('A')}")
print(f"_excel2num('Z') = {_excel2num('Z')}")
print(f"_excel2num('AA') = {_excel2num('AA')}")
print(f"_range2cols('A,B,C') = {_range2cols('A,B,C')}")
print(f"_range2cols('A:C') = {_range2cols('A:C')}")
```

<details>

<summary>
Output showing _excel2num returns -1 for empty strings
</summary>
```
Testing _excel2num with empty string:
_excel2num('') = -1

Testing _excel2num with whitespace-only string:
_excel2num('   ') = -1

Testing _range2cols with trailing comma:
_range2cols('A,') = [0, -1]

Testing _range2cols with only comma:
_range2cols(',') = [-1, -1]

Testing _range2cols with multiple trailing commas:
_range2cols('A,,') = [0, -1, -1]

Testing _range2cols with empty element between valid columns:
_range2cols('A,,B') = [0, -1, 1]

Testing valid inputs:
_excel2num('A') = 0
_excel2num('Z') = 25
_excel2num('AA') = 26
_range2cols('A,B,C') = [0, 1, 2]
_range2cols('A:C') = [0, 1, 2]
```
</details>

## Why This Is A Bug

The `_excel2num()` function violates its documented contract in multiple ways:

1. **Documentation states it should raise ValueError**: The docstring at lines 112-115 explicitly states "Raises ValueError: Part of the Excel column name was invalid." Empty strings and whitespace-only strings are clearly invalid Excel column names.

2. **Returns invalid column index**: The function returns `-1` for empty input, which is not a valid 0-based column index. This violates the function's return value contract (lines 107-110) which promises "The column index corresponding to the name."

3. **Algorithm incorrectly handles empty iteration**: The loop at line 119 `for c in x.upper().strip():` iterates zero times for an empty string. The function then returns `index - 1` where `index` is still 0, resulting in `-1`.

4. **Propagates to higher-level functions**: The bug causes `_range2cols()` to return lists containing `-1` values, which violates its promise to return "A list of 0-based column indices" (lines 141-142). This can affect `read_excel()`'s `usecols` parameter processing.

5. **Excel compatibility**: In Excel itself, empty column references trigger #REF! or #NAME? errors. The function should similarly reject invalid input rather than returning a value that appears valid.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/excel/_util.py`. The `_excel2num` function (lines 98-127) is used by `_range2cols` (lines 130-160), which in turn is called by `maybe_convert_usecols` (line 209) when processing string usecols arguments for `read_excel()`.

This affects any pandas users who:
- Use `read_excel()` with a string `usecols` parameter containing trailing commas
- Process user-provided column specifications that might contain formatting issues
- Rely on proper error handling for invalid column specifications

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

## Proposed Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -116,6 +116,10 @@ def _excel2num(x: str) -> int:
     """
     index = 0

+    # Strip whitespace and check for empty string
+    x = x.strip()
+    if not x:
+        raise ValueError(f"Invalid column name: empty string")
+
     for c in x.upper().strip():
         cp = ord(c)
```