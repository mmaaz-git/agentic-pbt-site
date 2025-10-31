# Bug Report: Pydantic Base64Bytes Silent Data Loss

**Target**: `pydantic.types.Base64Bytes`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Base64Bytes` type silently loses data when initialized with invalid base64-encoded bytes. Instead of raising a `ValidationError`, it accepts the invalid input and decodes it to empty bytes, causing silent data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.types import Base64Bytes

@given(st.binary(min_size=1))
def test_base64bytes_no_silent_data_loss(data):
    class Model(BaseModel):
        field: Base64Bytes

    try:
        m = Model(field=data)
        assert len(m.field) > 0 or len(data) == 0, \
            f"Silent data loss: {len(data)} bytes became {len(m.field)} bytes"
    except ValidationError:
        pass
```

**Failing input**: `data = b'\x00'` (or any invalid base64 bytes like `b'\x01\x02\x03'`, `b'\x80\xff\xfe'`)

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.types import Base64Bytes

class Model(BaseModel):
    data: Base64Bytes

m = Model(data=b'\x00')
print(f"Input: b'\\x00' (1 byte)")
print(f"Output: {m.data!r} (0 bytes)")
print(f"Silent data loss!")
```

Output:
```
Input: b'\x00' (1 byte)
Output: b'' (0 bytes)
Silent data loss!
```

## Why This Is A Bug

1. **Silent data corruption**: The field silently loses data without any warning or error. A user passing `b'\x00'` would expect either:
   - A `ValidationError` because it's not valid base64
   - The data to be stored as-is
   - NOT for it to become empty bytes

2. **Violates principle of least surprise**: When a validation library accepts input without error, users expect the data to be preserved.

3. **Root cause**: Python's `base64.b64decode()` silently returns empty bytes for certain invalid inputs when not using `validate=True`. Pydantic is calling `base64.b64decode()` without validation, leading to silent failures.

4. **Inconsistent behavior**: Some invalid base64 bytes (like `b'hello'`) raise `ValidationError`, while others (like `b'\x00'`) silently become empty bytes.

## Fix

The fix is to use strict base64 validation to catch invalid input:

```diff
# In pydantic's base64 validation code
def validate_base64(value: bytes) -> bytes:
    try:
-       return base64.b64decode(value)
+       return base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValidationError(...)
```

Alternatively, pydantic could add additional checks to ensure the decoded result is not unexpectedly empty:

```diff
def validate_base64(value: bytes) -> bytes:
    try:
        decoded = base64.b64decode(value)
+       # Verify that empty result is expected (empty input → empty output)
+       if len(decoded) == 0 and len(value) > 0:
+           # Try with validation to get proper error message
+           base64.b64decode(value, validate=True)
        return decoded
    except Exception as e:
        raise ValidationError(...)
```