# Bug Report: xarray.coding.strings - Null Byte Data Corruption in String Encoding/Decoding

**Target**: `xarray.coding.strings.encode_string_array` and `xarray.coding.strings.decode_bytes_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip encoding/decoding of string arrays silently corrupts data when strings contain specific null byte patterns, violating the documented VariableCoder contract that `coder.decode(coder.encode(variable)) == variable`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

@given(
    strings=arrays(
        dtype=object,
        shape=st.integers(min_value=1, max_value=20),
        elements=st.text(min_size=0, max_size=50)
    )
)
def test_string_encode_decode_roundtrip(strings):
    encoded = encode_string_array(strings, encoding='utf-8')
    decoded = decode_bytes_array(encoded, encoding='utf-8')
    assert np.array_equal(decoded, strings)

if __name__ == "__main__":
    test_string_encode_decode_roundtrip()
```

<details>

<summary>
**Failing input**: `strings=array(['\x00'], dtype=object)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 22, in <module>
    test_string_encode_decode_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 10, in test_string_encode_decode_roundtrip
    strings=arrays(
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 19, in test_string_encode_decode_roundtrip
    assert np.array_equal(decoded, strings)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_string_encode_decode_roundtrip(
    strings=array(['\x00'], dtype=object),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.coding.strings import encode_string_array, decode_bytes_array

# Test case 1: Single null byte
original = np.array(['\x00'], dtype=object)
print(f"Test 1: Single null byte")
print(f"Original: {original!r}")
encoded = encode_string_array(original)
print(f"Encoded: {encoded!r}")
decoded = decode_bytes_array(encoded)
print(f"Decoded: {decoded!r}")
print(f"Round-trip successful: {np.array_equal(decoded, original)}")
print()

# Test case 2: Null byte at beginning
original2 = np.array(['\x00hello'], dtype=object)
print(f"Test 2: Null byte at beginning")
print(f"Original: {original2!r}")
encoded2 = encode_string_array(original2)
print(f"Encoded: {encoded2!r}")
decoded2 = decode_bytes_array(encoded2)
print(f"Decoded: {decoded2!r}")
print(f"Round-trip successful: {np.array_equal(decoded2, original2)}")
print()

# Test case 3: Null byte in middle
original3 = np.array(['a\x00b'], dtype=object)
print(f"Test 3: Null byte in middle")
print(f"Original: {original3!r}")
encoded3 = encode_string_array(original3)
print(f"Encoded: {encoded3!r}")
decoded3 = decode_bytes_array(encoded3)
print(f"Decoded: {decoded3!r}")
print(f"Round-trip successful: {np.array_equal(decoded3, original3)}")
print()

# Test case 4: Null byte at end
original4 = np.array(['hello\x00'], dtype=object)
print(f"Test 4: Null byte at end")
print(f"Original: {original4!r}")
encoded4 = encode_string_array(original4)
print(f"Encoded: {encoded4!r}")
decoded4 = decode_bytes_array(encoded4)
print(f"Decoded: {decoded4!r}")
print(f"Round-trip successful: {np.array_equal(decoded4, original4)}")
```

<details>

<summary>
Data corruption with null bytes - two failure cases
</summary>
```
Test 1: Single null byte
Original: array(['\x00'], dtype=object)
Encoded: array([b''], dtype='|S1')
Decoded: array([''], dtype=object)
Round-trip successful: False

Test 2: Null byte at beginning
Original: array(['\x00hello'], dtype=object)
Encoded: array([b'\x00hello'], dtype='|S6')
Decoded: array(['\x00hello'], dtype=object)
Round-trip successful: True

Test 3: Null byte in middle
Original: array(['a\x00b'], dtype=object)
Encoded: array([b'a\x00b'], dtype='|S3')
Decoded: array(['a\x00b'], dtype=object)
Round-trip successful: True

Test 4: Null byte at end
Original: array(['hello\x00'], dtype=object)
Encoded: array([b'hello'], dtype='|S6')
Decoded: array(['hello'], dtype=object)
Round-trip successful: False
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of xarray's encoding/decoding system. The `VariableCoder` base class explicitly documents in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/coding/common.py:29-30` that subclasses should implement encode() and decode() methods that satisfy the identity: `coder.decode(coder.encode(variable)) == variable`.

The `EncodedStringCoder` class inherits from `VariableCoder` and uses `encode_string_array`/`decode_bytes_array` functions, but these functions fail to preserve this round-trip property for certain null byte patterns:

1. **Single null byte strings** (`'\x00'`) are corrupted to empty strings (`''`)
2. **Strings ending with null bytes** (`'hello\x00'`) have the trailing null byte silently removed (`'hello'`)
3. **Strings with null bytes in the middle or beginning** (when not a single character) are preserved correctly

This inconsistent behavior leads to silent data corruption without any warning or error, which is particularly dangerous in scientific computing contexts where data integrity is paramount.

## Relevant Context

The root cause lies in NumPy's handling of fixed-width byte string dtypes (e.g., `'S1'`, `'S6'`). When `encode_string_array` uses `np.array(encoded, dtype=bytes)` at line 103 of `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/coding/strings.py`, NumPy creates fixed-width byte arrays that follow C-style string conventions:

- A single null byte `b'\x00'` in a `dtype='S1'` array is stored as empty `b''`
- Trailing null bytes are treated as string terminators and stripped
- Embedded null bytes (not at the end) are preserved

This behavior is documented in NumPy's string dtype documentation and relates to how NumPy interfaces with C libraries that use null-terminated strings.

The issue affects any xarray operations that serialize/deserialize string data, particularly when working with netCDF files or other formats that use the `EncodedStringCoder`.

## Proposed Fix

```diff
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -100,7 +100,7 @@ def decode_bytes_array(bytes_array, encoding="utf-8"):
 def encode_string_array(string_array, encoding="utf-8"):
     string_array = np.asarray(string_array)
     encoded = [x.encode(encoding) for x in string_array.ravel()]
-    return np.array(encoded, dtype=bytes).reshape(string_array.shape)
+    return np.array(encoded, dtype=object).reshape(string_array.shape)
```

Using `dtype=object` instead of `dtype=bytes` avoids NumPy's fixed-width string dtype behavior and preserves all bytes including nulls. This ensures the round-trip property is maintained for all valid Unicode strings.