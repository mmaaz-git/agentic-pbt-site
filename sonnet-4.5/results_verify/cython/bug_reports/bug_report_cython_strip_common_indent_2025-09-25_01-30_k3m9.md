# Bug Report: Cython.Build.Inline.strip_common_indent Undefined Variable

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The function `strip_common_indent` uses an undefined variable `indent` in its second loop, which either causes a `NameError` when processing code with only comments/blank lines, or uses a stale value from a previous loop iteration leading to incorrect comment detection.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from Cython.Build.Inline import strip_common_indent

@given(st.lists(st.sampled_from(['#comment', '  #comment', '', '  '])))
@example(['#comment'])
@example(['  #comment'])
@example([''])
def test_strip_common_indent_only_comments_and_blanks(lines):
    code = '\n'.join(lines)
    result = strip_common_indent(code)
```

**Failing input**:
```python
  x = 1
    y = 2
 #comment
  z = 3
```

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

code = """  x = 1
    y = 2
 #comment
  z = 3"""

result = strip_common_indent(code)

result_lines = result.splitlines()
comment_line = result_lines[2]

assert comment_line == ' #comment'
```

The assertion fails. The comment line's leading space is incorrectly stripped.

## Why This Is A Bug

In the function's second loop (line 422 of Inline.py), the variable `indent` is used:

```python
if not match or not line or line[indent:indent+1] == '#':
```

However, `indent` is only defined in the **first loop** (line 415) and retains the value from the last iteration.

For the failing input:
1. First loop processes lines and sets `indent=2` (from last non-comment line 'z = 3')
2. Second loop checks if ` #comment` is a comment by testing `line[2:3] == '#'`
3. But ` #comment`[2] = `'c'`, not `'#'`
4. So the comment is incorrectly processed instead of being skipped
5. The leading space gets stripped, producing `#comment` instead of preserving ` #comment`

Comments at indentation levels different from the stale `indent` value are not correctly detected and thus incorrectly modified.

## Fix

```diff
--- a/Cython/Build/Inline.py
+++ b/Cython/Build/Inline.py
@@ -419,7 +419,7 @@ def strip_common_indent(code):
             min_indent = indent
     for ix, line in enumerate(lines):
         match = _find_non_space(line)
-        if not match or not line or line[indent:indent+1] == '#':
+        if not match or not line or (match and line[match.start():match.start()+1] == '#'):
             continue
         lines[ix] = line[min_indent:]
     return '\n'.join(lines)
```

Or more concisely:

```diff
-        if not match or not line or line[indent:indent+1] == '#':
+        if not match or not line or (match and line[match.start()] == '#'):
```