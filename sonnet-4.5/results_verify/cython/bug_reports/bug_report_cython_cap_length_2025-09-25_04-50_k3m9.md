# Bug Report: Cython.Compiler.PyrexTypes.cap_length Violates Length Constraint

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cap_length` function fails to enforce its length constraint when `max_len < 17`, returning strings longer than the specified maximum.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.PyrexTypes import cap_length


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)), st.integers(min_value=10, max_value=200))
def test_cap_length_respects_max(s, max_len):
    result = cap_length(s, max_len)
    assert len(result) <= max_len
```

**Failing input**: `s='00000000000'`, `max_len=10`

## Reproducing the Bug

```python
from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('00000000000', max_len=10)
print(f"Result: {result!r}")
print(f"Length: {len(result)}")
print(f"Expected max: 10")
```

Output:
```
Result: '9c9f57__0000__etc'
Length: 17
Expected max: 10
```

## Why This Is A Bug

The function is named `cap_length` and takes a `max_len` parameter, indicating it should cap strings at the specified maximum length. However, when `max_len < 17`, the function returns strings longer than `max_len`, violating this contract.

The bug occurs because the format string is `'{hash_prefix}__{s[:max_len-17]}__etc'` which has a fixed overhead of 13 characters. When `max_len - 17` is negative, the slice `s[:max_len-17]` includes more characters than expected, causing the result to exceed `max_len`.

## Fix

```diff
--- a/PyrexTypes.py
+++ b/PyrexTypes.py
@@ -1,5 +1,8 @@
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
+    if max_len < 17:
+        hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:max_len-4]
+        return '%s__etc' % hash_prefix
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
     return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```