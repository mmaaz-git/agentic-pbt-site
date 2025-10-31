# Bug Report: PyrexTypes.cap_length Violates Length Contract

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cap_length` function violates its contract by returning strings longer than `max_len` when the input string exceeds `max_len`. The function adds a fixed-size prefix and suffix that can exceed the specified maximum length.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from Cython.Compiler.PyrexTypes import cap_length

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
       st.integers(min_value=1, max_value=200))
@settings(max_examples=300)
def test_cap_length_respects_max(s, max_len):
    result = cap_length(s, max_len=max_len)
    assert len(result) <= max_len
```

**Failing input**: `s='00'`, `max_len=1`

## Reproducing the Bug

```python
from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('00', max_len=1)
assert len(result) <= 1

result2 = cap_length('a' * 100, max_len=10)
assert len(result2) <= 10
```

## Why This Is A Bug

The function name `cap_length` and parameter name `max_len` clearly indicate the function should cap the result to at most `max_len` characters. However, when the input exceeds `max_len`, the function creates a result with format `{hash}__{truncated}__etc` which has a minimum length of 13 characters (6-char hash + "__" + "" + "__etc").

This means:
- For `max_len < 13`, the function always returns strings longer than `max_len`
- For larger `max_len`, it can still exceed the limit if `max_len - 17 < 0`

## Fix

The function should ensure the total length including the hash prefix and suffix doesn't exceed `max_len`. One approach:

```diff
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
+
+    suffix = '__etc'
+    separator = '__'
+    min_overhead = 6 + len(separator) + len(suffix)
+
+    if max_len < min_overhead:
+        hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:max_len]
+        return hash_prefix
+
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
-    return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
+    truncated_len = max_len - min_overhead
+    return '%s__%s__etc' % (hash_prefix, s[:truncated_len])
```