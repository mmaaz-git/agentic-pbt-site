# Bug Report: scipy.io.arff split_data_line IndexError on Empty String Input

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function in scipy's ARFF reader crashes with an `IndexError` when given an empty string, attempting to access `line[-1]` without first checking if the string is non-empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line


@given(st.just(''))
def test_split_data_line_empty_string(line):
    result, dialect = split_data_line(line)

# Run the test
if __name__ == "__main__":
    test_split_data_line_empty_string()
```

<details>

<summary>
**Failing input**: `''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 11, in <module>
    test_split_data_line_empty_string()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_split_data_line_empty_string
    def test_split_data_line_empty_string(line):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 7, in test_split_data_line_empty_string
    result, dialect = split_data_line(line)
                      ~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
Falsifying example: test_split_data_line_empty_string(
    line='',
)
```
</details>

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

# Test with empty string
result, dialect = split_data_line('')
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/repo.py", line 4, in <module>
    result, dialect = split_data_line('')
                      ~~~~~~~~~~~~~~~^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

The `split_data_line` function is an internal utility used when parsing ARFF (Attribute-Relation File Format) data files. The function crashes with an uncaught `IndexError` when passed an empty string, which violates expected behavior in several ways:

1. **Empty strings can occur through normal code paths**:
   - At line 374 in `_arffread.py`, the function is called on results from `escaped_string.split("\n")`. If the string contains consecutive newlines (`\n\n`), the split operation produces empty strings between them.
   - At line 865, the function is called after filtering with the `r_empty` regex (defined as `re.compile(r'^\s+$')` at line 34), which only matches lines containing whitespace characters. Completely empty strings ('') do NOT match this regex and would pass through to `split_data_line`.

2. **The crash is clearly unintended**: The function attempts to check if the last character is a newline at line 476 (`if line[-1] == '\n':`) without first verifying the string is non-empty. This results in a low-level `IndexError` rather than any meaningful error handling.

3. **The code already attempts to handle edge cases**: The function strips whitespace after the newline check (line 480), and other parts of the codebase attempt to filter empty lines, suggesting they are expected edge cases that should be handled gracefully.

## Relevant Context

The `split_data_line` function is located in `/scipy/io/arff/_arffread.py` at lines 468-497. It's used to parse CSV-formatted data lines in ARFF files using Python's csv module. The function handles dialect detection for different delimiters and quoting styles.

Key code locations where empty strings could be passed to this function:
- Line 374: Processing relational attributes with potential double newlines
- Line 865: Main data parsing loop where empty line filtering may miss truly empty strings
- Line 130: Parsing nominal attribute values

The ARFF format specification doesn't explicitly state whether empty lines are allowed in data sections, but the existing code attempts to filter them, indicating they should be handled rather than causing crashes.

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