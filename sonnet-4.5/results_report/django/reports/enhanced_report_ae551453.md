# Bug Report: django.core.signing b62_decode Crashes on Empty String and Invalid Characters

**Target**: `django.core.signing.b62_decode`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `b62_decode` function crashes with `IndexError` when given an empty string and with `ValueError` when given invalid base62 characters like `:`, violating defensive programming principles for input validation.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.core.signing import b62_encode, b62_decode

@given(st.text())
def test_b62_decode_does_not_crash(s):
    try:
        result = b62_decode(s)
    except (IndexError, ValueError):
        assert False, f"b62_decode should not crash on input: {s!r}"

if __name__ == "__main__":
    test_b62_decode_does_not_crash()
```

<details>

<summary>
**Failing input**: `''` (empty string) and `':'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 15, in <module>
  |     test_b62_decode_does_not_crash()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 8, in test_b62_decode_does_not_crash
  |     def test_b62_decode_does_not_crash(s):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 10, in test_b62_decode_does_not_crash
    |     result = b62_decode(s)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/signing.py", line 85, in b62_decode
    |     decoded = decoded * 62 + BASE62_ALPHABET.index(digit)
    |                              ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    | ValueError: substring not found
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 12, in test_b62_decode_does_not_crash
    |     assert False, f"b62_decode should not crash on input: {s!r}"
    |            ^^^^^
    | AssertionError: b62_decode should not crash on input: ':'
    | Falsifying example: test_b62_decode_does_not_crash(
    |     s=':',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/28/hypo.py:11
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 10, in test_b62_decode_does_not_crash
    |     result = b62_decode(s)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/signing.py", line 80, in b62_decode
    |     if s[0] == "-":
    |        ~^^^
    | IndexError: string index out of range
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 12, in test_b62_decode_does_not_crash
    |     assert False, f"b62_decode should not crash on input: {s!r}"
    |            ^^^^^
    | AssertionError: b62_decode should not crash on input: ''
    | Falsifying example: test_b62_decode_does_not_crash(
    |     s='',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/28/hypo.py:11
    +------------------------------------
```
</details>

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

<details>

<summary>
IndexError on empty string, incorrect decoding of "-", and impact on TimestampSigner
</summary>
```
Bug 1: Crash on empty string
CRASH: IndexError - string index out of range

Bug 2: Invalid round-trip for '-'
b62_decode('-') = 0
b62_encode(0) = '0'
Round-trip: '-' -> 0 -> '0'
Expected: '-' should either raise an error or round-trip correctly

Bug 3: Impact on TimestampSigner
Handled: BadSignature
```
</details>

## Why This Is A Bug

The `b62_decode` function has multiple issues that violate expected behavior:

1. **IndexError on empty string**: At line 80 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/signing.py`, the code attempts to access `s[0]` without checking if the string is non-empty. This causes an `IndexError` instead of raising a meaningful error like `ValueError` with a descriptive message.

2. **ValueError on invalid characters**: The function uses `BASE62_ALPHABET.index(digit)` at line 85 which raises `ValueError: substring not found` when encountering non-base62 characters like `:`. While raising an error for invalid input is correct, the error message is cryptic and doesn't indicate the actual problem.

3. **Incorrect handling of standalone "-"**: When given "-" as input, the function returns 0, but `b62_encode(0)` returns "0", not "-". This violates the expected inverse relationship between encode and decode functions. The function strips the "-" sign but then has no digits to decode, resulting in 0.

4. **Impact on security components**: The `TimestampSigner.unsign()` method (line 269) calls `b62_decode` on potentially untrusted input from timestamps. While the current implementation catches these errors and raises `BadSignature`, the poor error handling could potentially be exploited or lead to confusing debugging scenarios.

## Relevant Context

The `b62_decode` function is an internal utility in Django's signing module, used primarily by `TimestampSigner` for encoding/decoding timestamps. The BASE62_ALPHABET consists of "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".

The function is designed to decode base62-encoded integers, with support for negative numbers (prefixed with "-"). However, it lacks proper input validation which leads to crashes on edge cases.

Code location: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/signing.py:76-86`

## Proposed Fix

```diff
--- a/django/core/signing.py
+++ b/django/core/signing.py
@@ -76,11 +76,18 @@ def b62_encode(s):
 def b62_decode(s):
     if s == "0":
         return 0
+    if not s:
+        raise ValueError("Cannot decode empty string")
     sign = 1
     if s[0] == "-":
         s = s[1:]
         sign = -1
+    if not s:
+        raise ValueError("Cannot decode standalone minus sign")
     decoded = 0
     for digit in s:
+        if digit not in BASE62_ALPHABET:
+            raise ValueError(f"Invalid base62 character: {digit!r}")
         decoded = decoded * 62 + BASE62_ALPHABET.index(digit)
     return sign * decoded
```