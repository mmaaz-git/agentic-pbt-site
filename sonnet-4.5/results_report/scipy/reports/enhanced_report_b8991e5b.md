# Bug Report: scipy.io.arff split_data_line IndexError on Empty String Input

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with an IndexError when processing an empty string, failing to check if the string is non-empty before accessing `line[-1]`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for scipy.io.arff.split_data_line function."""

from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text())
def test_split_data_line_handles_all_strings(line):
    """Test that split_data_line can handle any string input without crashing."""
    try:
        result, dialect = split_data_line(line)
        assert isinstance(result, list)
        print(f"✓ Passed for line: {repr(line)[:50]}")
    except ValueError:
        # ValueError is an expected exception for malformed data
        print(f"✓ ValueError (expected) for line: {repr(line)[:50]}")
    except IndexError as e:
        # IndexError should not happen
        print(f"✗ FAILED with IndexError for line: {repr(line)}")
        print(f"  Error: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis test on split_data_line function...")
    print("-" * 60)
    test_split_data_line_handles_all_strings()
    print("-" * 60)
    print("Test completed successfully!")
```

<details>

<summary>
**Failing input**: `''` (empty string)
</summary>
```
Running Hypothesis test on split_data_line function...
------------------------------------------------------------
✗ FAILED with IndexError for line: ''
  Error: string index out of range
✓ Passed for line: 'x@Y𢳭\U000a6288ù'
✓ Passed for line: '\U00109e24é'
✓ Passed for line: '\x96矬}\x8f'
✓ Passed for line: 'ÏApàÜØ×eR'
✓ Passed for line: 'Õ¶õ\U000b126b(Ùí\U000dbb17'
✓ Passed for line: '¿Ò\x1fZÀ`'
✓ Passed for line: ':Il8¿F𞊚\U0001d3ba\U000e4294¡'
✓ Passed for line: 'xb-\x0b'
✓ Passed for line: '\x9b\U0003cb7fö\U0003ec52=-\U00107365'
✗ FAILED with IndexError for line: ''
  Error: string index out of range
✗ FAILED with IndexError for line: ''
  Error: string index out of range
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 26, in <module>
    test_split_data_line_handles_all_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 8, in test_split_data_line_handles_all_strings
    def test_split_data_line_handles_all_strings(line):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 11, in test_split_data_line_handles_all_strings
    result, dialect = split_data_line(line)
                      ~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
Falsifying example: test_split_data_line_handles_all_strings(
    line='',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/63/hypo.py:14
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of scipy.io.arff.split_data_line IndexError on empty string."""

from scipy.io.arff._arffread import split_data_line

# Attempting to split an empty string
print("Testing split_data_line with empty string:")
try:
    result, dialect = split_data_line('')
    print(f"Success: result={result}, dialect={dialect}")
except IndexError as e:
    print(f"IndexError caught: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/repo.py", line 9, in <module>
    result, dialect = split_data_line('')
                      ~~~~~~~~~~~~~~~^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
Testing split_data_line with empty string:
IndexError caught: string index out of range
```
</details>

## Why This Is A Bug

The `split_data_line` function is designed to parse CSV-like data lines in ARFF files. It's called internally in three contexts within the module, most notably in `RelationalAttribute.parse_data()` at line 374, where it processes lines after splitting by newlines:

```python
for raw in escaped_string.split("\n"):
    row, self.dialect = split_data_line(raw, self.dialect)
```

When a string is split by newlines, empty strings are naturally produced in common scenarios:
- Data ending with a newline (e.g., `"data\n"` splits to `["data", ""]`)
- Multiple consecutive newlines (e.g., `"data\n\nmore"` splits to `["data", "", "more"]`)
- Data starting with a newline (e.g., `"\ndata"` splits to `["", "data"]`)

The function attempts to check if the last character is a newline (`line[-1] == '\n'`) at line 476 without first verifying the string is non-empty. In Python, accessing index -1 on an empty string raises an IndexError. This violates the principle of defensive programming - the function should handle edge cases gracefully, especially when called in contexts where empty strings are expected.

## Relevant Context

The ARFF (Attribute-Relation File Format) is a standard file format used by Weka and other machine learning tools for representing datasets. The `split_data_line` function is an internal utility that parses individual data lines, handling CSV-like formatting with support for various delimiters and quoting styles.

Key observations:
1. **Internal Function**: While `split_data_line` is not part of the public API (not listed in `__all__`), it's still critical for parsing relational attributes correctly.

2. **Natural Occurrence**: Empty strings aren't malformed input - they occur naturally when processing multi-line relational data, which is a supported ARFF feature.

3. **No Documentation Prohibition**: There's no documentation stating that empty strings are invalid input, and the ARFF specification doesn't forbid empty lines in data sections.

4. **Simple Fix**: The bug is a classic bounds-checking error with a straightforward one-line fix.

Links:
- Function location: `/scipy/io/arff/_arffread.py:468-497`
- Primary usage in relational parsing: `/scipy/io/arff/_arffread.py:373-374`
- ARFF format documentation: https://www.cs.waikato.ac.nz/ml/weka/arff.html

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