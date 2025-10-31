# Bug Report: Cython.Build.Inline strip_common_indent Comment Mangling

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `strip_common_indent` function mangles comment lines when they are indented differently than the last non-comment line, causing the '#' character to be stripped and the comment to become regular code.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from Cython.Build.Inline import strip_common_indent


@st.composite
def code_with_comments_at_different_indents(draw):
    code_indent = draw(st.integers(min_value=2, max_value=10))
    comment_indent = draw(st.integers(min_value=0, max_value=10))
    assume(comment_indent < code_indent)

    return (
        ' ' * code_indent + 'x = 1\n' +
        ' ' * comment_indent + '# comment\n' +
        ' ' * code_indent + 'y = 2'
    )


@given(code_with_comments_at_different_indents())
@settings(max_examples=500)
def test_strip_common_indent_preserves_comment_markers(code):
    """
    Property: Comment lines should retain their '#' marker.
    Evidence: Comments are syntactically significant in Python
    """
    result = strip_common_indent(code)

    for line in result.split('\n'):
        stripped_line = line.strip()
        if stripped_line and 'comment' in stripped_line:
            assert stripped_line.startswith('#'), \
                f"Comment line lost '#' marker: {line!r}"
```

**Failing input**:
```python
"""    x = 1
  # comment
    y = 2"""
```

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

test_input = """    x = 1
  # comment
    y = 2"""

result = strip_common_indent(test_input)

print("Input:")
print(test_input)
print("\nOutput:")
print(result)
```

**Expected output**:
```
x = 1
# comment
y = 2
```

**Actual output**:
```
x = 1
comment
y = 2
```

The '#' is incorrectly stripped from the comment line.

## Why This Is A Bug

The function uses two loops in `Inline.py:408-425`:

**First loop** (lines 411-419) finds the minimum indentation:
- Sets `indent = match.start()` for EVERY line, including comments (line 415)
- Skips comments via `continue` after updating `indent` (line 417)
- After the loop, `indent` contains the indent level from the last processed line

**Second loop** (lines 420-424) strips the minimum indentation:
- Line 422: `if not match or not line or line[indent:indent+1] == '#':`
- **BUG**: Uses stale `indent` from first loop instead of current line's indent
- When a comment has different indent than the last line from loop 1, the check fails
- Comment is not skipped and has `min_indent` stripped, mangling it

**Execution trace** for the example:
1. First loop: `indent` is 2 after processing comment line, then 4 after last code line
2. Second loop on comment line (`  # comment`):
   - Has match at position 2
   - Checks `line[4:5]` = `'c'` (not `'#'`!)
   - Comment is NOT skipped
   - Executes `line[min_indent:]` which strips the `'# '` prefix

## Fix

```diff
--- a/Cython/Build/Inline.py
+++ b/Cython/Build/Inline.py
@@ -419,7 +419,7 @@ def strip_common_indent(code):
             min_indent = indent
     for ix, line in enumerate(lines):
         match = _find_non_space(line)
-        if not match or not line or line[indent:indent+1] == '#':
+        if not match or not line or line[match.start():match.start()+1] == '#':
             continue
         lines[ix] = line[min_indent:]
     return '\n'.join(lines)
```

The fix uses `match.start()` to check the current line's first non-space character, rather than using the stale `indent` from the first loop.