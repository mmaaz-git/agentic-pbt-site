# Bug Report: pydantic.v1 bytes field JSON serialization crash

**Target**: `pydantic.v1.BaseModel.json()` with bytes fields
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Models with `bytes` fields crash when calling `.json()` if the bytes contain non-UTF-8 data, making JSON serialization unreliable for binary data.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel


class Model(BaseModel):
    data: bytes


@given(st.binary(min_size=0, max_size=100))
def test_bytes_field_roundtrip(data):
    m = Model(data=data)
    d = m.dict()
    json_str = m.json()

    recreated_from_dict = Model(**d)
    assert recreated_from_dict.data == data

    recreated_from_json = Model.parse_raw(json_str)
    assert recreated_from_json.data == data
```

**Failing input**: `b'\x80'` (and any non-UTF-8 bytes)

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel


class Model(BaseModel):
    data: bytes


m = Model(data=b'\x80')
print(f"Model created: {m}")

print("\nAttempting to serialize to JSON...")
try:
    json_str = m.json()
    print(f"Success: {json_str}")
except UnicodeDecodeError as e:
    print(f"Failed: {e}")
```

Output:
```
Model created: data=b'\x80'

Attempting to serialize to JSON...
Failed: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The `bytes` type is designed to hold arbitrary binary data, not just UTF-8 text. Pydantic v1's JSON encoder attempts to decode bytes as UTF-8 strings, which fails for non-UTF-8 byte sequences.

This is a serious bug because:

1. **Data type mismatch**: `bytes` should handle arbitrary binary data, but the JSON encoder silently assumes UTF-8
2. **Unpredictable failures**: Some byte values work (e.g., `b'hello'`), others crash (e.g., `b'\x80'`)
3. **No documentation**: Users aren't warned that `.json()` will fail for non-UTF-8 bytes
4. **Inconsistent behavior**: `.dict()` works fine, but `.json()` crashes
5. **Silent corruption risk**: UTF-8-compatible bytes succeed but may lose data

## Fix

The JSON encoder in pydantic v1 (located in `/pydantic/v1/json.py`) has this line:

```python
bytes: lambda o: o.decode(),
```

This should use base64 encoding instead:

```diff
- bytes: lambda o: o.decode(),
+ bytes: lambda o: b64encode(o).decode('ascii'),
```

And add the corresponding import:

```diff
+ from base64 import b64encode, b64decode
```

The decoder would also need to be updated to handle base64-encoded bytes when parsing.