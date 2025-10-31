# Bug Report: pydantic.deprecated.json bytes Encoder Crashes on Non-UTF-8 Bytes

**Target**: `pydantic.deprecated.json.pydantic_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pydantic_encoder` function crashes with `UnicodeDecodeError` when encoding bytes containing non-UTF-8 sequences. The bytes encoder calls `o.decode()` without specifying encoding or error handling.

## Property-Based Test

```python
import warnings
from hypothesis import given, strategies as st
from pydantic.deprecated.json import pydantic_encoder
import json

warnings.filterwarnings('ignore', category=DeprecationWarning)

@given(st.binary(min_size=1, max_size=100))
def test_pydantic_encoder_bytes(b):
    result = json.dumps(b, default=pydantic_encoder)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'` (or any bytes with invalid UTF-8 sequences)

## Reproducing the Bug

```python
import json
from pydantic.deprecated.json import pydantic_encoder

non_utf8_bytes = b'\x80\x81\x82'

result = json.dumps(non_utf8_bytes, default=pydantic_encoder)
```

**Output**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The `pydantic_encoder` function is designed to handle encoding of Python types for JSON serialization, including bytes. However, it assumes all bytes are valid UTF-8, which is not always true. Bytes can contain arbitrary binary data that may not be valid UTF-8.

The current implementation at line 55 of `pydantic/deprecated/json.py`:
```python
bytes: lambda o: o.decode(),
```

This calls `decode()` without specifying an encoding or error handling strategy, causing it to crash on non-UTF-8 bytes.

## Fix

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

This change makes the encoder handle non-UTF-8 bytes gracefully by replacing invalid characters with the Unicode replacement character (�) instead of crashing.