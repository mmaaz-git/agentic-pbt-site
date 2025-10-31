# Bug Report: pydantic.deprecated.json.pydantic_encoder Crashes on Non-UTF-8 Bytes

**Target**: `pydantic.deprecated.json.pydantic_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pydantic_encoder` function crashes with `UnicodeDecodeError` when encoding bytes objects that contain invalid UTF-8 sequences. The encoder assumes all bytes are UTF-8 encoded strings, but Python's `bytes` type can contain arbitrary binary data.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.deprecated.json import pydantic_encoder


@given(st.binary())
def test_pydantic_encoder_bytes_decode(b):
    result = pydantic_encoder(b)
    assert isinstance(result, str)
    assert result == b.decode()
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pydantic.deprecated.json import pydantic_encoder

invalid_utf8_bytes = b'\x80'

try:
    result = pydantic_encoder(invalid_utf8_bytes)
    print(f"Success: {result}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

Python's `bytes` type can contain arbitrary binary data, not just UTF-8 encoded strings. The encoder's bytes handler is defined as `lambda o: o.decode()`, which calls `bytes.decode()` without any encoding or error handling parameters. This defaults to UTF-8 decoding with 'strict' error handling, causing crashes on non-UTF-8 bytes.

While JSON doesn't have a native bytes type, the encoder should handle arbitrary bytes gracefully rather than crashing. Common use cases for bytes in Python include:
- Binary file contents
- Encrypted or compressed data
- Network protocol payloads
- Binary serialization formats

The crash is particularly problematic because the error message (`UnicodeDecodeError`) doesn't clearly indicate that the issue is with the JSON encoder's bytes handling.

## Fix

The encoder should use base64 encoding for bytes (matching common JSON serialization practices) or at minimum handle decoding errors gracefully:

```diff
 ENCODERS_BY_TYPE: dict[type[Any], Callable[[Any], Any]] = {
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: o.decode('utf-8', errors='replace'),
     Color: str,
     datetime.date: isoformat,
```

Or better, use base64 encoding which is a common standard for encoding binary data in JSON:

```diff
+import base64
+
 ENCODERS_BY_TYPE: dict[type[Any], Callable[[Any], Any]] = {
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: base64.b64encode(o).decode('ascii'),
     Color: str,
     datetime.date: isoformat,
```