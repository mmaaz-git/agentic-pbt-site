# Bug Report: ENCODERS_BY_TYPE bytes Encoder Crashes on Non-UTF-8 Bytes

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
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pydantic.deprecated.json import ENCODERS_BY_TYPE, pydantic_encoder

non_utf8_bytes = b'\x80'
encoder = ENCODERS_BY_TYPE[bytes]
result = encoder(non_utf8_bytes)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The encoder uses `lambda o: o.decode()` which defaults to UTF-8 decoding. However, Python's `bytes` type is designed to hold arbitrary binary data, not just UTF-8 text. Many legitimate use cases involve non-UTF-8 bytes:

- Binary file data (images, executables, etc.)
- Raw network protocol data
- Encrypted or compressed data
- Any byte sequence with values > 0x7F that isn't valid UTF-8

The encoder should handle all possible byte values (0x00-0xFF), but crashes on many common inputs.

## Fix

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

Alternatively, use latin-1 encoding which has a 1-to-1 mapping with byte values 0x00-0xFF:

```diff
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: o.decode('latin-1'),
```