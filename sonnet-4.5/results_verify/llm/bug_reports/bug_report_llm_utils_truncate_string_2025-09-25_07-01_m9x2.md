# Bug Report: llm.utils.truncate_string Violates Length Constraint

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than `max_length` when `max_length` is very small (< 3).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length_bound(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length
```

**Failing input**: `text='00', max_length=1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from llm.utils import truncate_string

result = truncate_string('00', max_length=1)
print(f"Input: '00', max_length=1")
print(f"Result: '{result}', length={len(result)}")
```

Output:
```
Input: '00', max_length=1
Result: '...', length=3
```

## Why This Is A Bug

The function's docstring states "Truncate a string to a **maximum length**" and has a parameter `max_length: Maximum length of the result string`. However, when `max_length < 3`, the function returns `"..."` (length 3), violating this constraint. The same issue occurs for `max_length=2`.

This happens because line 476 computes `text[:max_length - 3] + "..."` without checking if the ellipsis itself exceeds `max_length`.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,6 +473,9 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
+        if max_length < 3:
+            # Can't fit ellipsis, just truncate
+            return text[:max_length]
         return text[: max_length - 3] + "..."
```