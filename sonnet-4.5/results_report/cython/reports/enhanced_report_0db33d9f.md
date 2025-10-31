# Bug Report: Cython.Build.Inline strip_common_indent Comment Corruption

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `strip_common_indent` function corrupts comment lines by stripping the '#' character and initial content when comments have less indentation than previously processed code lines.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from Cython.Build.Inline import strip_common_indent


@st.composite
def code_with_comment_at_lower_indent(draw):
    code_indent = draw(st.integers(min_value=2, max_value=10))
    comment_indent = draw(st.integers(min_value=0, max_value=code_indent-1))
    assume(comment_indent < code_indent)

    return (
        ' ' * code_indent + 'x = 1\n' +
        ' ' * comment_indent + '# comment\n' +
        ' ' * code_indent + 'y = 2'
    )


@given(code_with_comment_at_lower_indent())
@settings(max_examples=500)
def test_strip_common_indent_preserves_comment_hash(code):
    result = strip_common_indent(code)
    for inp_line, out_line in zip(code.split('\n'), result.split('\n')):
        if inp_line.lstrip().startswith('#'):
            assert out_line.lstrip().startswith('#'), f"Comment line lost '#' character!\nInput: {repr(code)}\nOutput: {repr(result)}"


if __name__ == "__main__":
    test_strip_common_indent_preserves_comment_hash()
```

<details>

<summary>
**Failing input**: `'  x = 1\n# comment\n  y = 2'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 28, in <module>
    test_strip_common_indent_preserves_comment_hash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 19, in test_strip_common_indent_preserves_comment_hash
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 24, in test_strip_common_indent_preserves_comment_hash
    assert out_line.lstrip().startswith('#'), f"Comment line lost '#' character!\nInput: {repr(code)}\nOutput: {repr(result)}"
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AssertionError: Comment line lost '#' character!
Input: '  x = 1\n# comment\n  y = 2'
Output: 'x = 1\ncomment\ny = 2'
Falsifying example: test_strip_common_indent_preserves_comment_hash(
    code='  x = 1\n# comment\n  y = 2',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

test_input = """  x = 1
# comment
  y = 2"""

print("Input:")
print(repr(test_input))
print("\nFormatted Input:")
print(test_input)

result = strip_common_indent(test_input)

print("\nOutput:")
print(repr(result))
print("\nFormatted Output:")
print(result)

print("\n--- Analysis ---")
for i, (inp_line, out_line) in enumerate(zip(test_input.split('\n'), result.split('\n')), 1):
    print(f"Line {i}:")
    print(f"  Input:  {repr(inp_line)}")
    print(f"  Output: {repr(out_line)}")
    if inp_line.lstrip().startswith('#'):
        if not out_line.lstrip().startswith('#'):
            print(f"  ERROR: Comment line lost '#' character!")
```

<details>

<summary>
Comment '#' character and leading content incorrectly stripped
</summary>
```
Input:
'  x = 1\n# comment\n  y = 2'

Formatted Input:
  x = 1
# comment
  y = 2

Output:
'x = 1\ncomment\ny = 2'

Formatted Output:
x = 1
comment
y = 2

--- Analysis ---
Line 1:
  Input:  '  x = 1'
  Output: 'x = 1'
Line 2:
  Input:  '# comment'
  Output: 'comment'
  ERROR: Comment line lost '#' character!
Line 3:
  Input:  '  y = 2'
  Output: 'y = 2'
```
</details>

## Why This Is A Bug

The `strip_common_indent` function is intended to normalize indentation in Python code while preserving the structure and meaning. However, it corrupts comment lines, fundamentally changing the code's semantics by converting comments into code statements.

The bug occurs due to incorrect variable reuse in the implementation (lines 408-425 of Inline.py):

1. **First loop (lines 411-419)**: Iterates through all lines to find the minimum indentation level of non-comment lines. The variable `indent` is set for EVERY line (including blank lines and comments) to track each line's indentation, but only non-comment lines update `min_indent`.

2. **Second loop (lines 420-424)**: Attempts to strip the common indentation. Line 422 incorrectly uses the stale `indent` variable (which contains the indentation from the last line processed in the first loop) instead of the current line's actual indentation.

The problematic condition `line[indent:indent+1] == '#'` checks the wrong position for the '#' character. For the failing example:
- After the first loop, `indent = 2` (from the last line "  y = 2")
- When processing "# comment", it checks position 2 (which is 'c', not '#')
- The comment is incorrectly treated as regular code and has its first 2 characters stripped

This violates the expected behavior documented in the function's usage throughout Cython, where comments should be preserved as comments regardless of their indentation level relative to surrounding code.

## Relevant Context

The `strip_common_indent` function is used by:
- `cython_inline()` (line 179) - for inline Cython code compilation
- `cymeit()` (lines 326-327) - for benchmarking Cython code

Both functions are part of Cython's public API and are used to compile Python/Cython code snippets dynamically. The bug can cause:
1. **Syntax errors**: When comment text becomes invalid Python (e.g., "# TODO: fix" becomes "TODO: fix")
2. **Silent semantic changes**: When comment text happens to be valid Python (e.g., "# print" becomes "print")
3. **Confusing error messages**: Users see errors about code that was supposed to be comments

Source code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Inline.py:408-425`

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