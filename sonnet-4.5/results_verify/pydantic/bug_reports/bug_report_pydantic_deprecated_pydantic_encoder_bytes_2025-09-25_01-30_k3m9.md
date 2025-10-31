# Bug Report: pydantic.deprecated.json.pydantic_encoder UnicodeDecodeError on Non-UTF8 Bytes

**Target**: `pydantic.deprecated.json.pydantic_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pydantic_encoder` function crashes with a `UnicodeDecodeError` when encoding bytes that contain invalid UTF-8 sequences, which is a common occurrence for arbitrary binary data.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.deprecated.json import pydantic_encoder

@given(st.binary(min_size=0, max_size=100))
def test_pydantic_encoder_bytes(b):
    result = pydantic_encoder(b)
    assert isinstance(result, str)
    assert result == b.decode()
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.deprecated.json import pydantic_encoder

invalid_utf8_bytes = b'\x80'
result = pydantic_encoder(invalid_utf8_bytes)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The `pydantic_encoder` function is designed to encode Python objects for JSON serialization. Bytes objects can contain arbitrary binary data, not just UTF-8 valid text. The encoder should handle all valid bytes objects, not crash on certain byte sequences.

Many legitimate use cases involve non-UTF-8 bytes:
- Binary file data (images, PDFs, etc.)
- Encrypted data
- Serialized binary formats
- Network protocols that use raw bytes

## Fix

The encoder should use `errors='ignore'` or `errors='replace'` when decoding, or better yet, use base64 encoding for arbitrary bytes:

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -52,7 +52,7 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:


 ENCODERS_BY_TYPE: dict[type[Any], Callable[[Any], Any]] = {
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: o.decode('utf-8', errors='replace'),
     Color: str,
     datetime.date: isoformat,
     datetime.datetime: isoformat,
```

Alternatively, for a more robust solution that preserves the original bytes:

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -1,6 +1,7 @@
 import datetime
 import warnings
 from collections import deque
+import base64
 from decimal import Decimal
 from enum import Enum
 from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
@@ -52,7 +53,7 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:


 ENCODERS_BY_TYPE: dict[type[Any], Callable[[Any], Any]] = {
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: base64.b64encode(o).decode('ascii'),
     Color: str,
     datetime.date: isoformat,
     datetime.datetime: isoformat,
```