# Bug Report: Cython.Debugger.Tests source_to_lineno Loses Duplicate Lines

**Target**: `Cython.Debugger.Tests.TestLibCython.source_to_lineno`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `source_to_lineno` dictionary loses duplicate source lines, mapping all occurrences of a line to only the last occurrence's line number.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=50))
@settings(max_examples=1000)
def test_source_to_lineno_preserves_all_lines(lines):
    lines_with_duplicates = lines + [lines[0]]

    source_to_lineno = {line.strip(): i for i, line in enumerate(lines_with_duplicates, 1)}

    first_occurrence = 1
    last_occurrence = len(lines_with_duplicates)

    mapped_lineno = source_to_lineno[lines[0].strip()]

    assert mapped_lineno == first_occurrence or mapped_lineno == last_occurrence

    if len(lines) != len(set(line.strip() for line in lines)):
        assert mapped_lineno == last_occurrence, \
            "Dictionary comprehension maps duplicate keys to last value, losing earlier occurrences"
```

**Failing input**: Any file with duplicate lines (e.g., multiple "pass" statements)

## Reproducing the Bug

```python
test_file_content = """cpdef eggs():
    pass

cdef ham():
    pass

cdef class SomeClass(object):
    def spam(self):
        pass
"""

lines = test_file_content.strip().split('\n')
source_to_lineno = {line.strip(): i for i, line in enumerate(lines, 1)}

pass_occurrences = [i for i, line in enumerate(lines, 1) if line.strip() == 'pass']
print(f"'pass' appears on lines: {pass_occurrences}")
print(f"source_to_lineno['pass'] = {source_to_lineno['pass']}")

assert pass_occurrences == [2, 5, 9]
assert source_to_lineno['pass'] == 9
```

## Why This Is A Bug

The `source_to_lineno` dictionary is used by `break_and_run()` and `lineno_equals()` methods to convert source lines to line numbers for GDB breakpoints. When a source line appears multiple times in the file (like "pass"), the dictionary comprehension `{line.strip(): i for i, line in enumerate(f, 1)}` maps the line to only the last occurrence.

In the actual Cython test `codefile`, "pass" appears on lines 26, 29, and 33, but `source_to_lineno['pass']` only maps to line 33. This makes it impossible to set breakpoints on the earlier "pass" statements using `break_and_run('pass')`.

## Fix

Replace the dictionary comprehension with a different data structure that preserves all line occurrences, such as a dict mapping to lists of line numbers:

```diff
--- a/Cython/Debugger/Tests/TestLibCython.py
+++ b/Cython/Debugger/Tests/TestLibCython.py
@@ -21,7 +21,13 @@ root = os.path.dirname(os.path.abspath(__file__))
 codefile = os.path.join(root, 'codefile')
 cfuncs_file = os.path.join(root, 'cfuncs.c')

 with open(codefile) as f:
-    source_to_lineno = {line.strip(): i for i, line in enumerate(f, 1)}
+    source_to_lineno = {}
+    for i, line in enumerate(f, 1):
+        stripped = line.strip()
+        if stripped not in source_to_lineno:
+            source_to_lineno[stripped] = i
+        # Alternative: raise an error if duplicates are found
+        # or use a dict of lists: source_to_lineno.setdefault(stripped, []).append(i)
```

Alternatively, ensure the `codefile` has no duplicate lines, or modify `break_and_run()` to accept line numbers directly instead of source lines.