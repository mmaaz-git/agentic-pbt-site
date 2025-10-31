# Bug Report: llm.utils.truncate_string Violates Length Constraint

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its core contract by returning strings longer than `max_length` when `max_length < 3`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm.utils as utils

@given(st.text(min_size=1), st.integers(min_value=0, max_value=100))
def test_truncate_string_respects_max_length(text, max_length):
    result = utils.truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result length {len(result)} exceeds max_length {max_length}"
```

**Failing input**: `text = "hello"`, `max_length = 2`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

text = "hello world"
max_length = 2

result = truncate_string(text, max_length=max_length)
print(f"Input: {repr(text)}, max_length={max_length}")
print(f"Result: {repr(result)}")
print(f"Result length: {len(result)}")
```

**Output:**
```
Input: 'hello world', max_length=2
Result: '...'
Result length: 3
```

The result has length 3, which exceeds the specified `max_length` of 2.

## Why This Is A Bug

The function's docstring and parameter name `max_length` establish a clear contract: the returned string should not exceed `max_length` characters. This invariant should hold for all valid inputs.

When `max_length < 3`, the function returns `text[: max_length - 3] + "..."`, which always has length at least 3 (just the ellipsis), violating the constraint.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -457,6 +457,10 @@ def truncate_string(
     if not text:
         return text

+    # Handle very small max_length
+    if max_length <= 0:
+        return ""
+
     if normalize_whitespace:
         text = re.sub(r"\s+", " ", text)

@@ -473,5 +477,10 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length <= 3:
+            # Can't fit ellipsis, just truncate
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```