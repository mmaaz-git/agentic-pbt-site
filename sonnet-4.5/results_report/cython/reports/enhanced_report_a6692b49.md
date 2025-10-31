# Bug Report: Cython.Build.Inline.strip_common_indent Mangles Comments Due to Undefined Variable

**Target**: `Cython.Build.Inline.strip_common_indent`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `strip_common_indent` uses an out-of-scope variable `indent` in its second loop, causing comments to be incorrectly identified and mangled - both the '#' character and indentation are lost.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example, settings
from Cython.Build.Inline import strip_common_indent

# Test with various combinations of comments and blanks
@given(st.lists(st.sampled_from(['#comment', '  #comment', '', '  '])))
@example(['#comment'])
@example(['  #comment'])
@example([''])
@settings(max_examples=10, deadline=None)
def test_strip_common_indent_only_comments_and_blanks(lines):
    code = '\n'.join(lines)
    try:
        result = strip_common_indent(code)
        print(f"Input: {repr(code)}")
        print(f"Output: {repr(result)}")
    except Exception as e:
        print(f"Exception for input {repr(code)}: {e}")
        raise

# Test the specific failing case from the bug report
def test_specific_failing_case():
    print("\n=== Testing specific failing case ===")
    code = """  x = 1
    y = 2
 #comment
  z = 3"""

    print(f"Input code:\n{repr(code)}\n")
    result = strip_common_indent(code)
    print(f"Result:\n{repr(result)}\n")

    result_lines = result.splitlines()
    comment_line = result_lines[2]

    print(f"Comment line: {repr(comment_line)}")
    print(f"Expected: {repr(' #comment')}")

    try:
        assert comment_line == ' #comment'
        print("✓ Test passed")
    except AssertionError:
        print(f"✗ Test FAILED: Comment line was incorrectly modified")
        print(f"  Got: {repr(comment_line)}")
        print(f"  Expected: ' #comment' (with leading space preserved)")
        raise

if __name__ == "__main__":
    print("Running property-based tests...")
    try:
        test_strip_common_indent_only_comments_and_blanks()
        print("\nAll property-based tests passed.\n")
    except Exception as e:
        print(f"Property-based test failed: {e}\n")

    try:
        test_specific_failing_case()
    except AssertionError:
        print("\nThe specific test case demonstrates the bug.")
```

<details>

<summary>
**Failing input**: `'  x = 1\n    y = 2\n #comment\n  z = 3'`
</summary>
```
Running property-based tests...
Input: '#comment'
Output: '#comment'
Input: '  #comment'
Output: '  #comment'
Input: ''
Output: ''
Input: ''
Output: ''
Input: '#comment'
Output: '#comment'
Input: '\n  #comment\n\n  '
Output: '\n  #comment\n\n  '
Input: '  \n\n#comment\n\n  '
Output: '  \n\n#comment\n\n  '
Input: '  #comment'
Output: '  #comment'
Input: '  #comment\n  '
Output: '  #comment\n  '
Input: '#comment\n  #comment'
Output: '#comment\n  #comment'
Input: '  #comment\n\n  \n#comment\n'
Output: '  #comment\n\n  \n#comment'
Input: '  '
Output: '  '
Input: '  \n  #comment\n  \n  \n  #comment\n#comment'
Output: '  \n  #comment\n  \n  \n  #comment\n#comment'

All property-based tests passed.


=== Testing specific failing case ===
Input code:
'  x = 1\n    y = 2\n #comment\n  z = 3'

Result:
'x = 1\n  y = 2\ncomment\nz = 3'

Comment line: 'comment'
Expected: ' #comment'
✗ Test FAILED: Comment line was incorrectly modified
  Got: 'comment'
  Expected: ' #comment' (with leading space preserved)

The specific test case demonstrates the bug.
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Inline import strip_common_indent

# Test case that demonstrates the bug
code = """  x = 1
    y = 2
 #comment
  z = 3"""

print("Input code:")
print(repr(code))
print()

result = strip_common_indent(code)

print("Result:")
print(repr(result))
print()

result_lines = result.splitlines()
print("Result lines:")
for i, line in enumerate(result_lines):
    print(f"Line {i}: {repr(line)}")
print()

# Check the comment line
comment_line = result_lines[2]
print(f"Comment line (index 2): {repr(comment_line)}")
print(f"Expected: {repr(' #comment')}")

# This should fail - the comment's leading space is incorrectly stripped
try:
    assert comment_line == ' #comment', f"Expected ' #comment' but got {repr(comment_line)}"
    print("✓ Assertion passed (unexpected!)")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
```

<details>

<summary>
Comment line mangled: '#' character and indentation lost
</summary>
```
Input code:
'  x = 1\n    y = 2\n #comment\n  z = 3'

Result:
'x = 1\n  y = 2\ncomment\nz = 3'

Result lines:
Line 0: 'x = 1'
Line 1: '  y = 2'
Line 2: 'comment'
Line 3: 'z = 3'

Comment line (index 2): 'comment'
Expected: ' #comment'
✗ Assertion failed: Expected ' #comment' but got 'comment'
```
</details>

## Why This Is A Bug

The function `strip_common_indent` is designed to remove common leading indentation from multi-line code while preserving relative indentation and handling comments specially. However, it contains a critical programming error on line 422 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Inline.py`:

```python
if not match or not line or line[indent:indent+1] == '#':
```

The variable `indent` used here is out of scope - it was only defined within the first loop (line 415) and retains the value from the last iteration of that loop. This causes several problems:

1. **Incorrect comment detection**: The function checks the wrong position in the line to identify comments. For the test input, after processing the last non-comment line `'  z = 3'`, `indent=2`. When checking the comment line `' #comment'`, it looks at position 2 (which is 'c') instead of position 1 (which is '#').

2. **Data corruption**: When the comment is not properly detected, it gets processed as a regular line. The function then strips from `min_indent` (which is 2), resulting in `' #comment'[2:]` = `'comment'`. Both the '#' character and the leading space are lost.

3. **Violation of function intent**: The first loop explicitly skips comments (lines 416-417) when calculating minimum indent, showing clear intent that comments should receive special handling. The bug defeats this intent.

## Relevant Context

- **Function location**: `/Cython/Build/Inline.py`, line 408-425
- **Used by**: `cython_inline()` and `cython_compile()` - public API functions
- **Similar function**: `Cython/Compiler/TreeFragment.py` has a similar `strip_common_indent` with proper documentation
- **No documentation**: The buggy function lacks any docstring or API documentation
- **Python scope rules**: In Python, variables defined in a loop are accessible after the loop ends but retain their last value. This is why the code doesn't raise a `NameError` but instead uses a stale value.

The bug only manifests when:
1. There are both code lines and comment lines
2. Comments have different indentation than the last processed code line
3. The stale `indent` value points to a position that doesn't contain '#'

## Proposed Fix

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