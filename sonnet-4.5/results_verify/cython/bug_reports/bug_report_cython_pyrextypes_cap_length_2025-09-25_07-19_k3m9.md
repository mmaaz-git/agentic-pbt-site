# Bug Report: Cython.Compiler.PyrexTypes.cap_length Exceeds max_len

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cap_length` function fails to cap string length to `max_len` when `max_len < 17`, instead producing strings significantly longer than the specified limit.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Compiler import PyrexTypes


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
       st.integers(min_value=10, max_value=200))
@settings(max_examples=1000)
def test_cap_length_respects_max_len(s, max_len):
    result = PyrexTypes.cap_length(s, max_len)
    assert len(result) <= max_len
```

**Failing input**: `s='00000000000', max_len=10`

## Reproducing the Bug

```python
from Cython.Compiler import PyrexTypes

result = PyrexTypes.cap_length('00000000000', 10)
print(f"Result: {result!r}")
print(f"Length: {len(result)}")
print(f"Expected: <= 10")
print(f"Actual: {len(result)} > 10")
```

Output:
```
Result: '9c9f57__0000__etc'
Length: 17
Expected: <= 10
Actual: 17 > 10
```

## Why This Is A Bug

The function is named `cap_length` and accepts `max_len` as a parameter, implying it should cap the string to at most `max_len` characters. However, when `max_len < 17`, the formula `s[:max_len-17]` produces a negative slice index (e.g., `s[:10-17]` becomes `s[:-7]`), which in Python means "all but the last 7 characters". This causes the function to include far more of the original string than intended, violating the fundamental contract implied by the function name.

The current implementation:
```python
def cap_length(s, max_len=63):
    if len(s) <= max_len:
        return s
    hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
    return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```

Results in format: `{6 chars}__{truncated}__etc{5 chars}` = 13 + len(truncated) characters.

When max_len < 17, `truncated = s[:negative]` which takes most of the string rather than limiting it.

## Fix

```diff
--- a/PyrexTypes.py
+++ b/PyrexTypes.py
@@ -5704,5 +5704,8 @@ def type_identifier_from_declaration(decl, scope=None):
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
+    if max_len < 13:
+        return s[:max_len]
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
-    return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
+    available = max_len - 13
+    return '%s__%s__etc' % (hash_prefix, s[:available])
```

The fix ensures that:
1. For very small max_len (< 13), we just truncate the string
2. For max_len >= 13, we use the hash format with correct length calculation (13 fixed chars + available space)
3. The total length is always <= max_len