# Bug Report: django.core.signing b62_decode IndexError and Invalid Input Handling

**Target**: `django.core.signing.b62_decode`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `b62_decode` function in `django.core.signing` crashes with `IndexError` when given an empty string, and returns incorrect values for invalid inputs like `"-"`. This violates both defensive programming principles and the documented round-trip property between `b62_encode` and `b62_decode`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.signing import b62_encode, b62_decode

@given(st.text())
def test_b62_decode_does_not_crash(s):
    try:
        result = b62_decode(s)
    except (IndexError, ValueError):
        assert False, f"b62_decode should not crash on input: {s!r}"
```

**Failing input**: `""` (empty string)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.signing import b62_encode, b62_decode

print("Bug 1: Crash on empty string")
try:
    result = b62_decode("")
    print(f"Result: {result}")
except IndexError as e:
    print(f"CRASH: IndexError - {e}")

print("\nBug 2: Invalid round-trip for '-'")
result = b62_decode("-")
print(f"b62_decode('-') = {result}")
reencoded = b62_encode(result)
print(f"b62_encode({result}) = '{reencoded}'")
print(f"Round-trip: '-' -> {result} -> '{reencoded}'")
print(f"Expected: '-' should either raise an error or round-trip correctly")

print("\nBug 3: Impact on TimestampSigner")
from django.core.signing import TimestampSigner
from django.conf import settings
if not settings.configured:
    settings.configure(SECRET_KEY='test', SECRET_KEY_FALLBACKS=[])

signer = TimestampSigner()
signed = signer.sign("test")
parts = signed.split(':')
malformed = f"{parts[0]}::{parts[2]}"

try:
    result = signer.unsign(malformed)
except IndexError:
    print("CRASH: TimestampSigner.unsign() crashes on malformed signature with empty timestamp")
except Exception as e:
    print(f"Handled: {type(e).__name__}")
```

## Why This Is A Bug

1. **Crash on valid string input**: The function crashes with `IndexError` when given an empty string. Functions should either handle all string inputs or explicitly validate them and raise appropriate errors (e.g., `ValueError`).

2. **Silent incorrect behavior**: For input `"-"`, the function returns `0`, but `b62_encode(0)` returns `"0"`, violating the expected inverse relationship. The function should either:
   - Reject invalid inputs with a clear error message
   - Round-trip correctly for all inputs

3. **Security implications**: The bug affects `TimestampSigner.unsign()` which calls `b62_decode` on untrusted input (line 269 in signing.py). A malformed signature with an empty timestamp field can crash the application.

4. **Root cause**: Line 80 accesses `s[0]` without checking if `s` is non-empty:
   ```python
   if s[0] == "-":  # IndexError if s is ""
   ```

## Fix

```diff
--- a/django/core/signing.py
+++ b/django/core/signing.py
@@ -76,6 +76,9 @@ def b62_encode(s):
 def b62_decode(s):
     if s == "0":
         return 0
+    if not s:
+        raise ValueError("Cannot decode empty string")
+    if s == "-":
+        raise ValueError("Cannot decode standalone minus sign")
     sign = 1
     if s[0] == "-":
         s = s[1:]
```

Alternatively, for more robust validation:

```diff
--- a/django/core/signing.py
+++ b/django/core/signing.py
@@ -76,6 +76,10 @@ def b62_encode(s):
 def b62_decode(s):
     if s == "0":
         return 0
+    if not s or (s[0] == "-" and len(s) == 1):
+        raise ValueError(
+            f"Invalid base62 string: {s!r}"
+        )
     sign = 1
     if s[0] == "-":
         s = s[1:]
```

This ensures:
1. Empty strings are rejected with a clear error message
2. Invalid inputs like `"-"` are rejected
3. The round-trip property holds for all valid inputs
4. `TimestampSigner.unsign()` will raise `ValueError` instead of crashing with `IndexError` on malformed input