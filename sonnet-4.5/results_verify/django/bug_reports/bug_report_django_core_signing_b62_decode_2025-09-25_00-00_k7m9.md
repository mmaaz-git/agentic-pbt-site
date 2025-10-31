# Bug Report: django.core.signing.b62_decode Invalid Input Handling

**Target**: `django.core.signing.b62_decode`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `b62_decode` function accepts invalid base62 encoded strings (specifically `-` and `-0`) and silently returns 0, violating the decode-encode round-trip property that base encoding functions should satisfy.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.signing import b62_encode, b62_decode

@given(st.text(min_size=1).filter(lambda x: all(c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-' for c in x)))
def test_b62_decode_encode_roundtrip(s):
    decoded = b62_decode(s)
    re_encoded = b62_encode(decoded)
    assert re_encoded == s, \
        f"Decode-encode round-trip failed: '{s}' -> {decoded} -> '{re_encoded}'"
```

**Failing inputs**: `-` and `-0`

## Reproducing the Bug

```python
from django.core.signing import b62_encode, b62_decode

result = b62_decode('-')
print(f"b62_decode('-') = {result}")
print(f"b62_encode({result}) = '{b62_encode(result)}'")

result = b62_decode('-0')
print(f"b62_decode('-0') = {result}")
print(f"b62_encode({result}) = '{b62_encode(result)}'")
```

Output:
```
b62_decode('-') = 0
b62_encode(0) = '0'
b62_decode('-0') = 0
b62_encode(0) = '0'
```

The problem: `b62_encode(0)` always returns `'0'`, never `'-'` or `'-0'`. Therefore, these are invalid base62 encoded strings that should be rejected.

## Why This Is A Bug

1. **Violates round-trip property**: For valid encoded strings, `b62_encode(b62_decode(s))` should equal `s`. This fails for `-` and `-0`.

2. **Inconsistent with encoder**: `b62_encode` never produces `-` or `-0` as output, so `b62_decode` shouldn't accept them as input.

3. **Silent failure**: The function accepts malformed input without validation, which could mask bugs in calling code.

4. **Mathematical incorrectness**: While `-0 == 0` in mathematics, in a string encoding scheme, the string representation should be canonical. There should only be one encoding for zero: `'0'`.

## Fix

```diff
--- a/django/core/signing.py
+++ b/django/core/signing.py
@@ -73,6 +73,8 @@ def b62_decode(s):
     if s == "0":
         return 0
     sign = 1
     if s[0] == "-":
         s = s[1:]
+        if not s or s == "0":
+            raise ValueError(f"Invalid base62 string: '{('-' + s) if s else '-'}'")
         sign = -1
     decoded = 0
     for digit in s:
```

This fix ensures that:
- `-` raises `ValueError: Invalid base62 string: '-'`
- `-0` raises `ValueError: Invalid base62 string: '-0'`
- Empty strings after stripping the minus sign are rejected
- The round-trip property is preserved
- Invalid input is explicitly rejected rather than silently mishandled