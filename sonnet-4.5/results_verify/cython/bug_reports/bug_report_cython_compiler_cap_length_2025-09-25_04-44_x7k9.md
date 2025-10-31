# Bug Report: Cython.Compiler.PyrexTypes.cap_length Violates Length Constraint

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cap_length` function violates its implied contract by returning strings longer than `max_len` when `max_len < 17`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from Cython.Compiler.PyrexTypes import cap_length

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1),
       st.integers(min_value=1, max_value=200))
def test_result_length_bounded(s, max_len):
    result = cap_length(s, max_len)
    if len(s) > max_len:
        assert len(result) <= max_len
```

**Failing input**: `s='00', max_len=1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('00', 1)
print(f"cap_length('00', max_len=1) = {repr(result)}")
print(f"Length: {len(result)} (expected: <= 1)")

result2 = cap_length('x' * 100, 10)
print(f"cap_length('x'*100, max_len=10) = {repr(result2)}")
print(f"Length: {len(result2)} (expected: <= 10)")
```

Output:
```
cap_length('00', max_len=1) = 'f15343____etc'
Length: 13 (expected: <= 1)
cap_length('x'*100, max_len=10) = '9dd4e4____etc'
Length: 13 (expected: <= 10)
```

## Why This Is A Bug

The function signature accepts any `max_len` value but fails to enforce it when `max_len < 17`. The implementation uses format `{6-char-hash}__{prefix}__etc`, which has a minimum length of 13 characters. When `max_len < 17`, the prefix becomes empty (due to `s[:max_len-17]` with negative index), resulting in a 13-character string that exceeds max_len.

While all current usage in the codebase uses `max_len=63`, the function is:
1. A public API (no underscore prefix)
2. Has a parameter that accepts arbitrary integers
3. Has a name that implies it will cap at max_len

## Fix

Add a minimum bound check or adjust the format for small max_len values:

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -5703,6 +5703,8 @@ def type_identifier(type, pyrex=False):
     return safe

 def cap_length(s, max_len=63):
+    if max_len < 17:
+        return s[:max_len]  # For very small max_len, simple truncation
     if len(s) <= max_len:
         return s
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
```

Alternative fix with documentation:

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -5703,6 +5703,11 @@ def type_identifier(type, pyrex=False):
     return safe

 def cap_length(s, max_len=63):
+    """Cap string length to max_len characters.
+
+    Note: When max_len < 17, the result may exceed max_len due to
+    the hash-based abbreviation format (minimum 13 characters).
+    """
     if len(s) <= max_len:
         return s
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
```