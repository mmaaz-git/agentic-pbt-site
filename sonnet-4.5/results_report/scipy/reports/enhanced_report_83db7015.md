# Bug Report: scipy.io.arff split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with `IndexError: string index out of range` when called with an empty string because it attempts to access `line[-1]` without checking if the string is non-empty.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis that demonstrates the split_data_line bug.
"""

from scipy.io.arff._arffread import split_data_line, RelationalAttribute
from hypothesis import given, strategies as st


@given(st.text())
def test_split_data_line_handles_any_string(line):
    """
    Property: split_data_line should handle any string input without crashing.

    This fails on empty strings due to unchecked indexing.
    """
    try:
        row, dialect = split_data_line(line)
        assert isinstance(row, (list, tuple))
    except ValueError:
        pass


@given(st.text())
def test_relational_parse_data_no_crash(data_str):
    """
    Property: RelationalAttribute.parse_data should not crash.

    This can trigger the bug when data_str ends with newline or is empty,
    because split('\\n') produces empty strings.
    """
    attr = RelationalAttribute("test")
    attr.attributes = []

    try:
        attr.parse_data(data_str)
    except IndexError as e:
        if "string index out of range" in str(e):
            raise AssertionError(f"IndexError on split_data_line with data: {repr(data_str)}") from e
    except Exception:
        pass


if __name__ == "__main__":
    # Run the tests
    print("Running test_split_data_line_handles_any_string...")
    test_split_data_line_handles_any_string()

    print("\nRunning test_relational_parse_data_no_crash...")
    test_relational_parse_data_no_crash()
```

<details>

<summary>
**Failing input**: `''` (empty string)
</summary>
```
Running test_split_data_line_handles_any_string...
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 47, in <module>
  |     test_split_data_line_handles_any_string()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 11, in test_split_data_line_handles_any_string
  |     def test_split_data_line_handles_any_string(line):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 18, in test_split_data_line_handles_any_string
    |     row, dialect = split_data_line(line)
    |                    ~~~~~~~~~~~~~~~^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 495, in split_data_line
    |     row = next(csv.reader([line], dialect))
    | _csv.Error: new-line character seen in unquoted field - do you need to open the file with newline=''?
    | Falsifying example: test_split_data_line_handles_any_string(
    |     line='0\r0',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 18, in test_split_data_line_handles_any_string
    |     row, dialect = split_data_line(line)
    |                    ~~~~~~~~~~~~~~~^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    |     if line[-1] == '\n':
    |        ~~~~^^^^
    | IndexError: string index out of range
    | Falsifying example: test_split_data_line_handles_any_string(
    |     line='',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the scipy.io.arff split_data_line bug.
This demonstrates that split_data_line crashes with IndexError when given an empty string.
"""

from scipy.io.arff._arffread import split_data_line

# Test with empty string - this will crash with IndexError
print("Testing split_data_line with empty string...")
result = split_data_line("")
print(f"Result: {result}")
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Testing split_data_line with empty string...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/repo.py", line 11, in <module>
    result = split_data_line("")
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

The `split_data_line` function at line 476 in `_arffread.py` attempts to check if a line ends with a newline character:

```python
if line[-1] == '\n':
    line = line[:-1]
```

However, when `line` is an empty string, accessing `line[-1]` raises an `IndexError` because there are no characters to index.

This violates the expected behavior in multiple ways:

1. **Inconsistent error handling**: The main data parser explicitly handles empty lines (see line 862 in `_arffread.py` where `r_empty.match(raw)` skips empty lines), but `RelationalAttribute.parse_data` does not perform this check before calling `split_data_line`.

2. **Defensive programming violation**: The function assumes it will never receive empty input, but doesn't document this requirement or validate the input.

3. **Common real-world scenario**: This bug is triggered when `RelationalAttribute.parse_data` processes data containing trailing newlines or empty strings, which occurs naturally when using Python's `str.split('\n')` method:
   - `"data\n".split('\n')` produces `['data', '']`
   - `"".split('\n')` produces `['']`
   - `"\n".split('\n')` produces `['', '']`

## Relevant Context

The bug manifests in the following code path:

1. `RelationalAttribute.parse_data` (line 373) splits data by newlines:
   ```python
   for raw in escaped_string.split("\n"):
       row, self.dialect = split_data_line(raw, self.dialect)
   ```

2. When the input contains trailing newlines or is empty, `split("\n")` produces empty strings that are passed directly to `split_data_line`.

3. The function crashes instead of either:
   - Handling empty strings gracefully (returning empty row)
   - Raising a meaningful ValueError about invalid input

The codebase already defines `r_empty = re.compile(r'^\s+$')` (line 34) to match empty lines, and the main data generator skips them (line 862), showing that empty lines are expected to be filtered out. However, this filtering is not consistently applied throughout the codebase.

Documentation reference: The function has no docstring specifying input requirements, and the ARFF format specification does not explicitly forbid empty lines in relational attributes.

## Proposed Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -473,7 +473,7 @@ def split_data_line(line, dialect=None):
     csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

     # Remove the line end if any
-    if line[-1] == '\n':
+    if line and line[-1] == '\n':
         line = line[:-1]

     # Remove potential trailing whitespace
```