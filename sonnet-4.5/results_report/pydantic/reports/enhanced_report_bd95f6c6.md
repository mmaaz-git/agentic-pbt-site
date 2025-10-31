# Bug Report: pydantic.deprecated.json pydantic_encoder Crashes on Non-UTF-8 Bytes

**Target**: `pydantic.deprecated.json.pydantic_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pydantic_encoder` function crashes with `UnicodeDecodeError` when encoding bytes objects that contain non-UTF-8 sequences, preventing successful JSON serialization of valid Python bytes objects.

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

if __name__ == "__main__":
    test_pydantic_encoder_bytes()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 14, in <module>
    test_pydantic_encoder_bytes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 9, in test_pydantic_encoder_bytes
    def test_pydantic_encoder_bytes(b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 10, in test_pydantic_encoder_bytes
    result = json.dumps(b, default=pydantic_encoder)
  File "/home/npc/miniconda/lib/python3.13/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
          ~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/json/encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "/home/npc/miniconda/lib/python3.13/json/encoder.py", line 261, in iterencode
    return _iterencode(o, 0)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/deprecated/json.py", line 107, in pydantic_encoder
    return encoder(obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/deprecated/json.py", line 55, in <lambda>
    bytes: lambda o: o.decode(),
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_pydantic_encoder_bytes(
    b=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
import json
import warnings
from pydantic.deprecated.json import pydantic_encoder

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Test case: bytes with non-UTF-8 sequences
non_utf8_bytes = b'\x80\x81\x82'

try:
    result = json.dumps(non_utf8_bytes, default=pydantic_encoder)
    print(f"Success: {result}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
```

<details>

<summary>
UnicodeDecodeError when encoding non-UTF-8 bytes
</summary>
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

The `pydantic_encoder` function is designed to handle encoding of various Python types for JSON serialization, including the `bytes` type as evidenced by its inclusion in the `ENCODERS_BY_TYPE` dictionary at line 55 of `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/deprecated/json.py`. However, the implementation assumes all bytes objects contain valid UTF-8 text by calling `o.decode()` without specifying encoding parameters or error handling.

This violates the expected behavior because:
1. Python's `bytes` type can contain arbitrary binary data, not just UTF-8 encoded text
2. The encoder explicitly claims to support the `bytes` type but fails on valid bytes values
3. The crash prevents JSON serialization from completing, even though there are reasonable ways to encode arbitrary bytes (e.g., base64, hex, or UTF-8 with error replacement)
4. The documentation does not specify that bytes must be UTF-8 decodable, creating an undocumented constraint

## Relevant Context

The bug occurs in the `ENCODERS_BY_TYPE` dictionary at line 55 of `pydantic/deprecated/json.py`:
```python
bytes: lambda o: o.decode(),
```

When `o.decode()` is called without parameters, Python defaults to:
- UTF-8 encoding
- 'strict' error handling mode
- This causes an immediate crash on any byte sequence that isn't valid UTF-8

While the module is in the `deprecated` namespace (with warnings directing users to `pydantic_core.to_jsonable_python`), it is still part of the public API and should handle all valid input gracefully rather than crashing.

Common approaches for handling arbitrary bytes in JSON include:
- Base64 encoding (most common for binary data)
- Hex string representation
- UTF-8 decoding with error replacement for text-like data

## Proposed Fix

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

This fix uses UTF-8 decoding with the 'replace' error handler, which substitutes the Unicode replacement character (�) for any invalid byte sequences. This ensures the encoder never crashes while maintaining backward compatibility for valid UTF-8 bytes.