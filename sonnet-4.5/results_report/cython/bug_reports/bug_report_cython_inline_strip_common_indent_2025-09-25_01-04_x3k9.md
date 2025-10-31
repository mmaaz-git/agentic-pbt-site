# Bug Report: Cython.Build.Inline strip_common_indent Comment Mangling

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `strip_common_indent` function mangles comment lines when they are indented differently than the last processed non-comment line, causing the '#' character and leading content to be incorrectly stripped.

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
            assert out_line.lstrip().startswith('#')
```

**Failing input**: `"  x = 1\n# comment\n  y = 2"`

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

test_input = """  x = 1
# comment
  y = 2"""

result = strip_common_indent(test_input)
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

## Why This Is A Bug

In `Inline.py:408-425`, the function uses a variable `indent` from the first loop in the second loop's condition check:

**First loop** (lines 411-419): Finds minimum indentation
```python
for line in lines:
    match = _find_non_space(line)
    if not match:
        continue
    indent = match.start()        # Set for EVERY line
    if line[indent] == '#':       # Skip comments for min_indent
        continue
    if min_indent is None or min_indent > indent:
        min_indent = indent
```
After this loop, `indent` holds the indent from the last processed line.

**Second loop** (lines 420-424): Strips indentation
```python
for ix, line in enumerate(lines):
    match = _find_non_space(line)
    if not match or not line or line[indent:indent+1] == '#':  # BUG HERE
        continue
    lines[ix] = line[min_indent:]
```
Line 422 uses the stale `indent` value instead of the current line's indent position (`match.start()`).

**For the failing input**:
- First loop: `indent` becomes 2 (from last code line)
- Second loop on comment line `# comment`:
  - `match.start()` = 0 (where '#' is)
  - But checks `line[2:3]` = `'c'` (not '#')
  - Comment is NOT skipped, gets processed as `'# comment'[2:]` = `'comment'`

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