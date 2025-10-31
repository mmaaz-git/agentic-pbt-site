# Bug Report: Cython.Tempita isolate_expression Bounds Checking

**Target**: `Cython.Tempita._tempita.isolate_expression`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `isolate_expression` function raises an `IndexError` when start position is past the end of the text, despite code comments indicating that positions past the end can occur.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita._tempita import isolate_expression

@given(st.text(min_size=1, max_size=50),
       st.integers(min_value=0, max_value=10))
def test_isolate_expression_positions_past_end(text, row_offset):
    lines = text.splitlines(True)
    num_lines = len(lines)

    start_row = num_lines + 1 + row_offset
    end_row = start_row

    result = isolate_expression(text, (start_row, 0), (end_row, 0))
    assert isinstance(result, str)
```

**Failing input**: `text='a', start_row=2, end_row=2`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import isolate_expression

text = "line1\nline2"
result = isolate_expression(text, (3, 0), (3, 0))
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

The function has a comment on line 1025 acknowledging that positions can be past the end of lines: "It'll sometimes give (end_row_past_finish, 0)". The code handles this case when `srow != erow` (lines 1024-1026), but fails to handle it when `srow == erow` (line 1021) or when `srow >= len(lines)` in the multiline case (line 1022), causing an IndexError.

The inconsistent bounds checking violates the defensive programming pattern established elsewhere in the function.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -1018,7 +1018,13 @@ def isolate_expression(string, start_pos, end_pos):
     erow -= 1
     lines = string.splitlines(True)
     if srow == erow:
-        return lines[srow][scol:ecol]
+        if srow < len(lines):
+            return lines[srow][scol:ecol]
+        else:
+            return ''
+    if srow >= len(lines):
+        return ''
     parts = [lines[srow][scol:]]
     parts.extend(lines[srow+1:erow])
     if erow < len(lines):
```