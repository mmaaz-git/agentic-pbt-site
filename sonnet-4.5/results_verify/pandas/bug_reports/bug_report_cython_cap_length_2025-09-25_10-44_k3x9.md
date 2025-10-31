# Bug Report: Cython.Compiler.PyrexTypes.cap_length Unicode Crash

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cap_length` function crashes with `UnicodeEncodeError` when given a string containing non-ASCII characters that exceeds `max_len`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Compiler import PyrexTypes as PT

@given(st.text())
@settings(max_examples=500)
def test_cap_length_respects_max(s):
    capped = PT.cap_length(s, max_len=63)
    assert len(capped) <= 63
```

**Failing input**: `'000000000000000000000000000000000000000000000000000000000000000\x80'` (64 characters: 63 zeros + one non-ASCII character)

## Reproducing the Bug

```python
from Cython.Compiler import PyrexTypes as PT

s = '000000000000000000000000000000000000000000000000000000000000000\x80'
result = PT.cap_length(s, max_len=63)
```

## Why This Is A Bug

The `cap_length` function is a public function (no leading underscore) that doesn't document requiring ASCII-only input. When the input string length exceeds `max_len` and contains non-ASCII characters, the function attempts to hash the string using `s.encode('ascii')` on line 5707, which raises `UnicodeEncodeError`.

While internal call sites may only pass ASCII strings, the function is public and should handle arbitrary Unicode strings gracefully.

## Fix

```diff
--- a/PyrexTypes.py
+++ b/PyrexTypes.py
@@ -5704,7 +5704,7 @@ def cap_length(s, max_len=63):
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
-    hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
+    hash_prefix = hashlib.sha256(s.encode('utf-8')).hexdigest()[:6]
     return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```