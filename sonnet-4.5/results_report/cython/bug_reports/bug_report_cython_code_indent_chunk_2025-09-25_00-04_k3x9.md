# Bug Report: Cython.Compiler.Code._indent_chunk Loses Single Characters

**Target**: `Cython.Compiler.Code._indent_chunk`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_indent_chunk` function incorrectly converts single-character strings (without trailing newlines) into newline characters, causing content loss. This affects strings of exactly length 1, such as '0', 'a', 'x', etc.

## Property-Based Test

```python
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
    assert original_chars == result_chars
```

**Failing input**: `s='0'`, `indent_len=0` (or any indentation value)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/cython')

from Cython.Compiler.Code import _indent_chunk

result = _indent_chunk('0', 0)
print(f"Result: {repr(result)}")

assert result == '\n'
```

Expected: `'0'` (content preserved)
Actual: `'\n'` (content lost)

This also affects other single characters:
```python
assert _indent_chunk('a', 0) == '\n'
assert _indent_chunk('x', 4) == '\n'
assert _indent_chunk('9', 0) == '\n'
```

However, multi-character strings work correctly:
```python
assert _indent_chunk('ab', 0) == 'ab'
assert _indent_chunk('abc', 0) == 'abc'
```

## Why This Is A Bug

The function's docstring states it should "Normalise leading space to the intended indentation and strip empty lines." Single-character strings are not empty lines and should have their content preserved.

The existing test suite includes `test_indent_one_line` which expects `_indent_chunk('abc', 0) == 'abc'`, demonstrating that single-line inputs should preserve content. Single characters are a valid edge case of single-line inputs.

The bug occurs at line 3318 in Code.py:
```python
if line_indentation + 1 == len(line):
    lines[i] = '\n'
```

This condition is meant to detect whitespace-only lines ending with a newline (e.g., "   \n"). However, for a single character like '0':
- `line_indentation = 0` (no leading spaces)
- `len(line) = 1` (just the character)
- `0 + 1 == 1` evaluates to `True`

Thus, single characters incorrectly match the "empty line" condition.

## Fix

The condition should check if the line actually ends with a newline before treating it as whitespace-only:

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

This ensures that only actual whitespace-only lines ending with newlines are stripped, while preserving single-character content.