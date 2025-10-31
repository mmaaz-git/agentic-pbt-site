# Bug Report: pydantic_encoder Crashes on Non-UTF8 Bytes

**Target**: `pydantic.deprecated.json.pydantic_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pydantic_encoder` function crashes with `UnicodeDecodeError` when encoding bytes objects that contain invalid UTF-8 sequences. The encoder uses `lambda o: o.decode()` which assumes all bytes are valid UTF-8, but this is not always the case for arbitrary binary data.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.deprecated.json import pydantic_encoder

@given(st.binary())
def test_pydantic_encoder_bytes(bytes_value):
    result = pydantic_encoder(bytes_value)
    assert isinstance(result, str)
    assert result == bytes_value.decode()
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pydantic.deprecated.json import pydantic_encoder

bytes_value = b'\x80'
result = pydantic_encoder(bytes_value)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The `pydantic_encoder` function is designed to convert Python objects into JSON-serializable representations. While bytes containing valid UTF-8 text should decode to strings, arbitrary binary data (such as image data, encrypted content, or other non-text bytes) is a valid use case.

The current implementation in `json.py:55` uses:
```python
bytes: lambda o: o.decode(),
```

This assumes all bytes objects contain valid UTF-8, which violates the documented behavior of JSON encoding in standard libraries (which typically use base64 encoding for binary data) and can crash on valid input.

## Fix

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -1,4 +1,5 @@
 import datetime
+import base64
 import warnings
 from collections import deque
 from decimal import Decimal
@@ -52,7 +53,7 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:


 ENCODERS_BY_TYPE: dict[type[Any], Callable[[Any], Any]] = {
-    bytes: lambda o: o.decode(),
+    bytes: lambda o: o.decode('utf-8', errors='replace'),
     Color: str,
     datetime.date: isoformat,
     datetime.datetime: isoformat,
```

**Note**: The fix uses `errors='replace'` to handle invalid UTF-8 gracefully. Alternatively, `errors='ignore'` or base64 encoding could be used depending on the desired behavior. Since this is deprecated code, a minimal fix that prevents crashes is most appropriate.