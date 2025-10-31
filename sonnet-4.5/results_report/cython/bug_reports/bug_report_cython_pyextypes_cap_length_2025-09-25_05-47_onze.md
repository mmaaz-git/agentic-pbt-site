# Bug Report: Cython.Compiler.PyrexTypes.cap_length Violates Length Contract

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cap_length()` function returns strings that exceed the specified `max_len` parameter when `max_len < 13`, violating its implied contract to cap string length.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Compiler.PyrexTypes import cap_length


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'), st.integers(min_value=0, max_value=200))
@settings(max_examples=500)
def test_cap_length_honors_max_len(s, max_len):
    result = cap_length(s, max_len)
    assert len(result) <= max_len
```

**Failing input**: `s='0'`, `max_len=0`

## Reproducing the Bug

```python
from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('0', max_len=0)
print(f"Result: {repr(result)}")
print(f"Result length: {len(result)}")
print(f"Expected max length: 0")
```

**Output**:
```
Result: '5feceb____etc'
Result length: 13
Expected max length: 0
```

## Why This Is A Bug

The function `cap_length(s, max_len=63)` has a parameter named `max_len` and a function name that implies capping the length of string `s` at `max_len`. However, when `len(s) > max_len`, the function constructs a result string using the format:

```python
'%s__%s__etc' % (hash_prefix, s[:max_len-17])
```

Where `hash_prefix` is 6 characters. This format has a minimum length of `6 + 2 + 0 + 5 = 13` characters (hash + "__" + string slice + "__etc"). Therefore, when `max_len < 13`, the function **always** returns a string longer than `max_len`, violating its API contract.

While all current callers in the codebase use the default `max_len=63`, the function accepts `max_len` as a parameter and should honor it for all valid values.

## Fix

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -5704,6 +5704,9 @@ def type_identifier(type, pyrex=False):
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
+    # Minimum format is "HASH__X__etc" (13 chars). For smaller max_len, just truncate.
+    if max_len < 13:
+        return s[:max_len]
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
     return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```