# Bug Report: Cython.Build.Inline strip_common_indent Comment Marker Stripping

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `strip_common_indent` function incorrectly strips the '#' character from comment lines when those comments have different indentation than the last non-comment line in the code, converting comments into invalid Python code.

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


if __name__ == "__main__":
    test_strip_common_indent_preserves_comment_markers()
```

<details>

<summary>
**Failing input**: `'  x = 1\n# comment\n  y = 2'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 35, in <module>
    test_strip_common_indent_preserves_comment_markers()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 19, in test_strip_common_indent_preserves_comment_markers
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 30, in test_strip_common_indent_preserves_comment_markers
    assert stripped_line.startswith('#'), \
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AssertionError: Comment line lost '#' marker: 'comment'
Falsifying example: test_strip_common_indent_preserves_comment_markers(
    code='  x = 1\n# comment\n  y = 2',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/50/hypo.py:30
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

test_input = """    x = 1
  # comment
    y = 2"""

result = strip_common_indent(test_input)

print("Input:")
print(repr(test_input))
print("\nOutput:")
print(repr(result))
print("\n--- Formatted Input ---")
print(test_input)
print("\n--- Formatted Output ---")
print(result)
```

<details>

<summary>
Comment marker '#' is stripped from the comment line
</summary>
```
Input:
'    x = 1\n  # comment\n    y = 2'

Output:
'x = 1\ncomment\ny = 2'

--- Formatted Input ---
    x = 1
  # comment
    y = 2

--- Formatted Output ---
x = 1
comment
y = 2
```
</details>

## Why This Is A Bug

The function violates the fundamental expectation that stripping common indentation should preserve the syntactic structure of Python/Cython code. Comment markers '#' are syntactically significant in Python and their removal transforms valid comments into invalid code statements.

The bug stems from a logic error in the function's implementation at `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Inline.py:408-425`. The function uses two loops:

1. **First loop (lines 411-419)**: Iterates through all lines to find the minimum indentation among non-comment lines. However, the variable `indent` is updated for EVERY line at line 415 (including comment lines), before the comment check at line 416-417. After this loop completes, `indent` contains the indentation level of the last processed line, not the current line being processed.

2. **Second loop (lines 420-424)**: Attempts to strip the minimum indentation from non-comment lines. Line 422 checks if a line is a comment using: `line[indent:indent+1] == '#'`. This is incorrect because `indent` contains the stale value from the last line of the first loop, not the current line's indentation. When a comment has different indentation than this stale value, the check fails to identify it as a comment.

For the failing example `'    x = 1\n  # comment\n    y = 2'`:
- First loop: `indent` becomes 4 (from the last line `'    y = 2'`)
- Second loop on comment line `'  # comment'`:
  - The actual first non-space character is at position 2
  - But the code checks position `[4:5]` which is `'c'` (not `'#'`)
  - The comment is not recognized and has its first 2 characters stripped
  - Result: `'comment'` instead of `'# comment'`

## Relevant Context

The `strip_common_indent` function is used internally by `cython_inline()` to normalize indentation in inline Cython code before compilation. The function has no documentation (no docstring or comments), but its name and usage context clearly indicate it should only remove common leading whitespace while preserving code structure.

The function explicitly attempts to handle comments specially (checks for '#' in both loops), showing clear intent to preserve comment markers. However, the implementation incorrectly reuses a variable from the first loop in the second loop's comment detection logic.

This bug can affect any Cython code using `cython_inline()` where comments have different indentation levels than surrounding code - a common occurrence in real-world code with explanatory comments at various indentation levels.

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Inline.py:408-425`

## Proposed Fix

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