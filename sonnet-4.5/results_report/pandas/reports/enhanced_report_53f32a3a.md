# Bug Report: pandas.io.excel._util._excel2num Returns -1 for Whitespace Instead of Raising ValueError

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns `-1` for whitespace-only or empty strings instead of raising `ValueError` as documented, violating the function's contract and potentially causing silent failures in downstream code.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for _excel2num whitespace bug"""

import string
from hypothesis import given, strategies as st, example
from pandas.io.excel._util import _excel2num
import pytest

@given(st.text(min_size=0, max_size=5))
@example(' ')  # single space
@example('')   # empty string
@example('\t') # tab
@example('\n') # newline
def test_excel2num_invalid_chars_raise_error(text):
    """_excel2num should raise ValueError for invalid characters"""
    # If the text contains any non-alphabetic character or is empty after stripping
    if not text.strip() or not all(c.isalpha() for c in text.strip()):
        with pytest.raises(ValueError, match="Invalid column name"):
            _excel2num(text)
    else:
        # Valid Excel column names should not raise
        result = _excel2num(text)
        assert isinstance(result, int)
        assert result >= 0

if __name__ == "__main__":
    # Run the test
    test_excel2num_invalid_chars_raise_error()
```

<details>

<summary>
**Failing input**: ` ` (single space), `''` (empty string), `'\t'` (tab), `'\n'` (newline)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 28, in <module>
  |     test_excel2num_invalid_chars_raise_error()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 10, in test_excel2num_invalid_chars_raise_error
  |     @example(' ')  # single space
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | BaseExceptionGroup: Hypothesis found 4 distinct failures in explicit examples. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 18, in test_excel2num_invalid_chars_raise_error
    |     with pytest.raises(ValueError, match="Invalid column name"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_excel2num_invalid_chars_raise_error(
    |     text=' ',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 18, in test_excel2num_invalid_chars_raise_error
    |     with pytest.raises(ValueError, match="Invalid column name"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_excel2num_invalid_chars_raise_error(
    |     text='',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 18, in test_excel2num_invalid_chars_raise_error
    |     with pytest.raises(ValueError, match="Invalid column name"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_excel2num_invalid_chars_raise_error(
    |     text='\t',
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 18, in test_excel2num_invalid_chars_raise_error
    |     with pytest.raises(ValueError, match="Invalid column name"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_excel2num_invalid_chars_raise_error(
    |     text='\n',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrate the _excel2num whitespace bug"""

from pandas.io.excel._util import _excel2num

# Test various whitespace inputs
test_cases = [
    ' ',      # single space
    '',       # empty string
    '\t',     # tab
    '\n',     # newline
    '   ',    # multiple spaces
    '\t\n',   # tab and newline
]

print("Testing whitespace inputs that should raise ValueError:")
print("=" * 60)

for test_input in test_cases:
    repr_input = repr(test_input)
    try:
        result = _excel2num(test_input)
        print(f"Input: {repr_input:<15} -> Result: {result} (BUG: should raise ValueError)")
    except ValueError as e:
        print(f"Input: {repr_input:<15} -> Raised ValueError: {e}")

print("\n" + "=" * 60)
print("For comparison, testing valid and invalid inputs:")
print("=" * 60)

# Test valid inputs
valid_cases = ['A', 'Z', 'AA', 'AB', 'XYZ']
for test_input in valid_cases:
    try:
        result = _excel2num(test_input)
        print(f"Input: {repr(test_input):<15} -> Result: {result} (valid)")
    except ValueError as e:
        print(f"Input: {repr(test_input):<15} -> Raised ValueError: {e}")

print()

# Test other invalid inputs that correctly raise errors
invalid_cases = ['A B', '123', 'A1', '!@#']
for test_input in invalid_cases:
    try:
        result = _excel2num(test_input)
        print(f"Input: {repr(test_input):<15} -> Result: {result}")
    except ValueError as e:
        print(f"Input: {repr(test_input):<15} -> Raised ValueError: {e} (correct)")
```

<details>

<summary>
Returns -1 for all whitespace inputs instead of raising ValueError
</summary>
```
Testing whitespace inputs that should raise ValueError:
============================================================
Input: ' '             -> Result: -1 (BUG: should raise ValueError)
Input: ''              -> Result: -1 (BUG: should raise ValueError)
Input: '\t'            -> Result: -1 (BUG: should raise ValueError)
Input: '\n'            -> Result: -1 (BUG: should raise ValueError)
Input: '   '           -> Result: -1 (BUG: should raise ValueError)
Input: '\t\n'          -> Result: -1 (BUG: should raise ValueError)

============================================================
For comparison, testing valid and invalid inputs:
============================================================
Input: 'A'             -> Result: 0 (valid)
Input: 'Z'             -> Result: 25 (valid)
Input: 'AA'            -> Result: 26 (valid)
Input: 'AB'            -> Result: 27 (valid)
Input: 'XYZ'           -> Result: 16899 (valid)

Input: 'A B'           -> Raised ValueError: Invalid column name: A B (correct)
Input: '123'           -> Raised ValueError: Invalid column name: 123 (correct)
Input: 'A1'            -> Raised ValueError: Invalid column name: A1 (correct)
Input: '!@#'           -> Raised ValueError: Invalid column name: !@# (correct)
```
</details>

## Why This Is A Bug

The `_excel2num` function has a clear contract documented in its docstring (lines 99-116 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/excel/_util.py`):

1. **Purpose**: Convert Excel column names like 'AB' to 0-based column indices
2. **Raises**: "ValueError - Part of the Excel column name was invalid"

The bug violates this contract in several ways:

- **Empty strings and whitespace are not valid Excel column names**: Excel column names must contain only letters A-Z. Whitespace characters, tabs, newlines, and empty strings cannot form valid column identifiers.

- **Returns invalid index instead of raising exception**: The function returns `-1` for these invalid inputs. Since column indices are 0-based non-negative integers, `-1` is not a valid column index and could cause array access errors or other bugs in downstream code.

- **Inconsistent error handling**: The function correctly raises `ValueError` for other invalid inputs like 'A B' (contains space), '123' (numeric), 'A1' (alphanumeric), but fails to do so for whitespace-only strings.

- **Root cause**: At line 119, the function calls `x.upper().strip()` which converts whitespace-only strings to empty strings. When the string is empty, the for loop never executes, leaving `index = 0`, and the function returns `0 - 1 = -1`.

## Relevant Context

The function is used internally by pandas for Excel file operations to convert column letters to numeric indices. When reading Excel files with `pandas.read_excel()`, column names are converted using this function. Returning `-1` instead of raising an error could lead to:

- Silent data corruption when invalid column references are used
- Hard-to-debug errors in code that assumes valid non-negative indices
- Potential security issues if negative indices are used for array access without bounds checking

The pandas documentation for Excel I/O operations doesn't mention any special handling for whitespace column names, reinforcing that they should be treated as invalid input.

Function location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/excel/_util.py:98-127`

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
    index = 0

-   for c in x.upper().strip():
+   stripped = x.upper().strip()
+   if not stripped:
+       raise ValueError(f"Invalid column name: {x}")
+
+   for c in stripped:
        cp = ord(c)

        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")

        index = index * 26 + cp - ord("A") + 1

    return index - 1
```