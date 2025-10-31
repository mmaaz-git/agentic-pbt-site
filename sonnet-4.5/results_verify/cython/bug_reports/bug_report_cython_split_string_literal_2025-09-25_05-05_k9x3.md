# Bug Report: Cython.Compiler.StringEncoding.split_string_literal Infinite Loop

**Target**: `Cython.Compiler.StringEncoding.split_string_literal`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_string_literal` function enters an infinite loop when given a string of backslashes with a small `limit` parameter (limit < 6). The function fails to make forward progress, causing the program to hang indefinitely.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(
    num_backslashes=st.integers(min_value=1, max_value=100),
    suffix=st.text(min_size=0, max_size=20),
    limit=st.integers(min_value=5, max_value=50)
)
@settings(max_examples=1000)
def test_split_string_literal_backslash_boundary(num_backslashes, suffix, limit):
    s = '\\' * num_backslashes + suffix
    result = split_string_literal(s, limit=limit)
    reconstructed = result.replace('""', '')
    assert reconstructed == s
```

**Failing input**: `s='\\\\\\\\' (4 backslashes), limit=2`

## Reproducing the Bug

```python
from Cython.Compiler.StringEncoding import split_string_literal

result = split_string_literal('\\\\\\\\', limit=2)
```

This call hangs indefinitely.

## Why This Is A Bug

The function is designed to split long string literals for MSVC compatibility (which has a 2000-character limit). However, when `limit` is small and the string contains many consecutive backslashes, the function's logic for avoiding splitting within escape sequences causes it to calculate an `end` position that doesn't advance past `start`, creating an infinite loop in the outer while loop.

The problematic code is at lines 314-319 in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/StringEncoding.py`:

```python
while s[end-1] == '\\':
    end -= 1
    if end == start:
        end = start + limit - (limit % 2) - 4
        break
```

When `limit < 6`, the calculation `start + limit - (limit % 2) - 4` can result in `end <= start`:
- For `limit=2`: `end = start + 2 - 0 - 4 = start - 2`
- For `limit=3`: `end = start + 3 - 1 - 4 = start - 2`
- For `limit=4`: `end = start + 4 - 0 - 4 = start`
- For `limit=5`: `end = start + 5 - 1 - 4 = start`

This causes the outer loop to never terminate.

## Fix

```diff
--- a/Cython/Compiler/StringEncoding.py
+++ b/Cython/Compiler/StringEncoding.py
@@ -313,7 +313,11 @@ def split_string_literal(s, limit=2000):
                 end -= 4 - s[end-4:end].find('\\')  # just before the backslash
                 while s[end-1] == '\\':
                     end -= 1
                     if end == start:
                         # must have been a long line of backslashes
-                        end = start + limit - (limit % 2) - 4
+                        # Ensure we always make forward progress
+                        end = start + max(1, limit - (limit % 2) - 4)
                         break
             chunks.append(s[start:end])
             start = end
```

This ensures that `end > start`, guaranteeing forward progress even with very small limits.