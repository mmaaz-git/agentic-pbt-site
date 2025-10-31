# Bug Report: Cython.Compiler.Code._indent_chunk Loses Single-Character Content

**Target**: `Cython.Compiler.Code._indent_chunk`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_indent_chunk` function incorrectly converts single-character strings (without trailing newlines) into newline characters, causing complete content loss. This affects any single character input like '0', 'a', 'x', etc.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings
from Cython.Compiler.Code import _indent_chunk

@given(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=200),
       st.integers(min_value=0, max_value=16))
@settings(max_examples=1000)
def test_indent_chunk_preserves_nonwhitespace(s, indent_len):
    assume('\t' not in s)
    result = _indent_chunk(s, indent_len)
    original_chars = ''.join(s.split())
    result_chars = ''.join(result.split())
    assert original_chars == result_chars, f"Content was lost: original={repr(s)}, result={repr(result)}"

if __name__ == "__main__":
    test_indent_chunk_preserves_nonwhitespace()
```

<details>

<summary>
**Failing input**: `s='0', indent_len=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 18, in <module>
    test_indent_chunk_preserves_nonwhitespace()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 8, in test_indent_chunk_preserves_nonwhitespace
    st.integers(min_value=0, max_value=16))
            ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 15, in test_indent_chunk_preserves_nonwhitespace
    assert original_chars == result_chars, f"Content was lost: original={repr(s)}, result={repr(result)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Content was lost: original='0', result='\n'
Falsifying example: test_indent_chunk_preserves_nonwhitespace(
    s='0',
    indent_len=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Code import _indent_chunk

# Test single character without newline
result = _indent_chunk('0', 0)
print(f"_indent_chunk('0', 0) = {repr(result)}")
print(f"Expected: '0', Got: {repr(result)}")
print(f"Content lost: {result != '0'}")
print()

# Test other single characters
print("Other single character tests:")
for char in ['a', 'x', '9', '1', 'b', 'z']:
    result = _indent_chunk(char, 0)
    print(f"_indent_chunk('{char}', 0) = {repr(result)} (expected: '{char}')")

print()

# Test with different indentation levels
print("Single character with various indentation:")
for indent in [0, 1, 4, 8]:
    result = _indent_chunk('0', indent)
    expected = ' ' * indent + '0'
    print(f"_indent_chunk('0', {indent}) = {repr(result)} (expected: {repr(expected)})")

print()

# Show that multi-character strings work correctly
print("Multi-character strings (working correctly):")
for s in ['ab', 'abc', '12', '123']:
    result = _indent_chunk(s, 0)
    print(f"_indent_chunk('{s}', 0) = {repr(result)} (expected: '{s}')")
```

<details>

<summary>
Single-character strings are incorrectly converted to newlines
</summary>
```
_indent_chunk('0', 0) = '\n'
Expected: '0', Got: '\n'
Content lost: True

Other single character tests:
_indent_chunk('a', 0) = '\n' (expected: 'a')
_indent_chunk('x', 0) = '\n' (expected: 'x')
_indent_chunk('9', 0) = '\n' (expected: '9')
_indent_chunk('1', 0) = '\n' (expected: '1')
_indent_chunk('b', 0) = '\n' (expected: 'b')
_indent_chunk('z', 0) = '\n' (expected: 'z')

Single character with various indentation:
_indent_chunk('0', 0) = '\n' (expected: '0')
_indent_chunk('0', 1) = '\n' (expected: ' 0')
_indent_chunk('0', 4) = '\n' (expected: '    0')
_indent_chunk('0', 8) = '\n' (expected: '        0')

Multi-character strings (working correctly):
_indent_chunk('ab', 0) = 'ab' (expected: 'ab')
_indent_chunk('abc', 0) = 'abc' (expected: 'abc')
_indent_chunk('12', 0) = '12' (expected: '12')
_indent_chunk('123', 0) = '123' (expected: '123')
```
</details>

## Why This Is A Bug

The function's docstring states it should "Normalise leading space to the intended indentation and strip empty lines." Single-character strings are not empty lines and should have their content preserved, not erased entirely.

The bug occurs at line 3318 in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/Code.py`. The problematic condition `if line_indentation + 1 == len(line):` is intended to detect whitespace-only lines that end with a newline (e.g., "   \n" where the line has 3 spaces and a newline, so indentation=3 and len=4).

However, for a single character like '0':
- `line_indentation = 0` (no leading spaces, as returned by `_count_indentation`)
- `len(line) = 1` (just the single character)
- The condition `0 + 1 == 1` evaluates to `True`
- The line is incorrectly replaced with '\n', losing the actual content

This violates the function's contract because:
1. Single characters are valid non-empty content
2. The function should only strip lines that consist entirely of whitespace
3. The existing behavior breaks the principle that non-whitespace content should be preserved

## Relevant Context

The `_indent_chunk` function processes text by:
1. Splitting it into lines using `splitlines(keepends=True)` to preserve newline characters
2. Identifying and normalizing indentation
3. Stripping lines that contain only whitespace

The bug affects line 3318-3319 where the condition incorrectly identifies single-character strings as empty lines. The condition was designed to match patterns like " \n", "  \n", "   \n" where `line_indentation + 1 == len(line)` because the line consists of spaces followed by a newline. However, it fails for the edge case of single characters without newlines.

This is an internal function (indicated by the underscore prefix) used in Cython's code generation pipeline. While single-character lines are uncommon in generated code, they are valid inputs that should be handled correctly.

## Proposed Fix

```diff
--- a/Cython/Compiler/Code.py
+++ b/Cython/Compiler/Code.py
@@ -3315,7 +3315,7 @@ def _indent_chunk(chunk: str, indentation_length: cython.int) -> str:
     i: cython.int
     for i, line in enumerate(lines):
         line_indentation = _count_indentation(line)
-        if line_indentation + 1 == len(line):
+        if line_indentation + 1 == len(line) and line.endswith('\n'):
             lines[i] = '\n'
         elif line_indentation < min_indentation:
             min_indentation = line_indentation
```