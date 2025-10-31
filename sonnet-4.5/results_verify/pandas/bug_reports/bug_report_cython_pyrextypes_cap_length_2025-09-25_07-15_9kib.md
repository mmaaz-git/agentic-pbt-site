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
print(f"Bug: {len(result)} > 10")
```

Output:
```
Result: '9c9f57__0000__etc'
Length: 17
Expected: <= 10
Bug: 17 > 10
```

## Why This Is A Bug

The function is named `cap_length` and accepts `max_len` as a parameter, implying it should cap the string to at most `max_len` characters. However, when `max_len < 17`, the formula `s[:max_len-17]` produces a negative slice index, causing the function to include far more of the original string than intended. This violates the fundamental contract implied by the function name and parameter.

While all current call sites use the default `max_len=63`, the function's public interface allows arbitrary `max_len` values, making this a latent bug that could affect future callers.

## Fix

```diff
--- a/PyrexTypes.py
+++ b/PyrexTypes.py
@@ -5703,5 +5703,7 @@ def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
-    return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
+    # Format: "hash__truncated__etc" = 6 + 2 + truncated + 5 = 13 + truncated
+    available_for_original = max(0, max_len - 13)
+    return '%s__%s__etc' % (hash_prefix, s[:available_for_original])
```
