# Bug Report: Cython.Build.Inline strip_common_indent Comment Corruption

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `strip_common_indent` function uses a stale loop variable when identifying comment lines, causing comments to be corrupted or completely removed when processing code with certain indentation patterns.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from Cython.Build.Inline import strip_common_indent


@given(st.integers(min_value=1, max_value=20), st.text(alphabet='abcdef', min_size=1, max_size=10))
@settings(max_examples=1000)
@example(indent=8, comment_text='comment')
@example(indent=5, comment_text='x')
def test_comment_preservation_with_code(indent, comment_text):
    code = ' ' * indent + 'code1\n#' + comment_text + '\n' + ' ' * indent + 'code2'

    result = strip_common_indent(code)
    result_lines = result.splitlines()

    assert len(result_lines) == 3
    comment_line = result_lines[1]
    assert comment_line.startswith('#'), f"Comment line should start with #, got: {repr(comment_line)}"
    assert comment_text in comment_line
```

**Failing input**: `indent=8, comment_text='comment'`

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

code = """        code1
#comment
        code2"""

result = strip_common_indent(code)

print("Input:")
print(code)
print("\nResult:")
print(result)
print("\nResult lines:")
for i, line in enumerate(result.splitlines()):
    print(f"  {i}: {repr(line)}")

expected_lines = ['code1', '#comment', 'code2']
actual_lines = result.splitlines()

assert actual_lines[1] == '#comment'
```

**Output**: The assertion fails. The middle line becomes `''` (empty string) instead of `'#comment'`. The comment is completely lost.

## Why This Is A Bug

The function's purpose is to strip common indentation from code blocks while preserving comment structure. Comments should retain their '#' prefix and content after indentation is removed.

The bug occurs because:
1. The first loop (lines 410-418) finds `min_indent` by examining non-comment lines
2. Variable `indent` is set during this loop to track the position of the first non-space character
3. The second loop (lines 420-424) reuses the stale `indent` variable from the first loop
4. Line 422 checks `line[indent:indent+1] == '#'` using this stale value
5. When `indent` doesn't match the comment's position, the comment fails the check
6. The comment line gets processed as code: `lines[ix] = line[min_indent:]`
7. When `min_indent` >= length of comment, this truncates or removes the entire comment

This violates the function's contract to preserve comments while stripping indentation.

## Fix

```diff
--- a/Cython/Build/Inline.py
+++ b/Cython/Build/Inline.py
@@ -419,7 +419,7 @@ def strip_common_indent(code):
             min_indent = indent
     for ix, line in enumerate(lines):
         match = _find_non_space(line)
-        if not match or not line or line[indent:indent+1] == '#':
+        if not match or not line or (match and line[match.start()] == '#'):
             continue
         lines[ix] = line[min_indent:]
     return '\n'.join(lines)
```

The fix replaces the stale `indent` variable with `match.start()`, which correctly identifies the position of the first non-space character in the current line being processed.