# Bug Report: pydantic.deprecated.json.ENCODERS_BY_TYPE[bytes] Crashes on Non-UTF-8 Bytes

**Target**: `pydantic.deprecated.json.ENCODERS_BY_TYPE[bytes]`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The bytes encoder in `ENCODERS_BY_TYPE` crashes with `UnicodeDecodeError` when encoding bytes that contain non-UTF-8 data, even though Python's `bytes` type is designed to hold arbitrary binary data.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.deprecated.json import ENCODERS_BY_TYPE


@given(st.binary())
@settings(max_examples=500)
def test_bytes_encoder(b):
    encoder = ENCODERS_BY_TYPE[bytes]
    result = encoder(b)
    assert isinstance(result, str)

if __name__ == "__main__":
    test_bytes_encoder()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 13, in <module>
    test_bytes_encoder()
    ~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 6, in test_bytes_encoder
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 9, in test_bytes_encoder
    result = encoder(b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/deprecated/json.py", line 55, in <lambda>
    bytes: lambda o: o.decode(),
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_bytes_encoder(
    b=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
from pydantic.deprecated.json import ENCODERS_BY_TYPE

# Test case: non-UTF-8 bytes
non_utf8_bytes = b'\x80'
encoder = ENCODERS_BY_TYPE[bytes]

try:
    result = encoder(non_utf8_bytes)
    print(f"Success: {result!r}")
except Exception as e:
    print(f"{e.__class__.__name__}: {e}")
```

<details>

<summary>
UnicodeDecodeError on non-UTF-8 byte
</summary>
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

The encoder uses `lambda o: o.decode()` which defaults to UTF-8 decoding. This violates expected behavior because:

1. **Python's bytes type accepts arbitrary binary data (0x00-0xFF)**, not just UTF-8 text. The type system makes no guarantee that bytes contain valid UTF-8.

2. **No documentation states this UTF-8 requirement**. The encoder is registered for the `bytes` type without any documented constraint that the bytes must be UTF-8 decodable.

3. **Common legitimate use cases fail**: Binary file data, encrypted data, compressed data, network protocols, and any bytes with values ≥ 0x80 that don't form valid UTF-8 sequences will crash.

4. **The crash is ungraceful** - it raises an unhandled `UnicodeDecodeError` rather than providing a fallback encoding strategy or clear error message about the limitation.

## Relevant Context

- The issue is in `/home/npc/miniconda/lib/python3.13/site-packages/pydantic/deprecated/json.py` line 55
- The module is marked as deprecated with warnings suggesting migration to `pydantic_core.to_jsonable_python`
- The JSON standard doesn't define how to encode binary data, but common practice is base64 encoding
- Python's standard `json` module refuses to encode bytes at all (raises TypeError)
- Even the modern pydantic v2 has similar UTF-8 assumptions for bytes serialization

Documentation: https://docs.pydantic.dev/latest/api/deprecated/
Source code: https://github.com/pydantic/pydantic/blob/main/pydantic/deprecated/json.py

## Proposed Fix

Use base64 encoding for bytes, which is the standard way to represent arbitrary binary data as JSON-safe strings:

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -1,5 +1,6 @@
 import datetime
 import warnings
+import base64
 from collections import deque
 from decimal import Decimal
 from enum import Enum
@@ -52,7 +53,7 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:


 ENCODERS_BY_TYPE: dict[type[Any], Callable[[Any], Any]] = {
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: base64.b64encode(o).decode('ascii'),
     Color: str,
     datetime.date: isoformat,
     datetime.datetime: isoformat,
```