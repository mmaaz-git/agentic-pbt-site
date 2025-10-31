# Bug Report: xarray.coding.strings - Null Byte Corruption in encode_string_array/decode_bytes_array

**Target**: `xarray.coding.strings.encode_string_array` and `xarray.coding.strings.decode_bytes_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property `decode_bytes_array(encode_string_array(strings)) == strings` is violated for strings containing null bytes (`'\x00'`). Null bytes are silently truncated, resulting in data corruption.

## Property-Based Test

```python
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
```

**Failing input**: `strings=array(['\x00'], dtype=object)`

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

original = np.array(['\x00'], dtype=object)
print(f"Original: {original!r}")

encoded = encode_string_array(original)
decoded = decode_bytes_array(encoded)

print(f"Decoded: {decoded!r}")
print(f"Round-trip successful: {np.array_equal(decoded, original)}")
```

**Output:**
```
Original: array(['\x00'], dtype=object)
Decoded: array([''], dtype=object)
Round-trip successful: False
```

The null byte is corrupted to an empty string.

**Additional failing examples:**
- `'a\x00b'` → `'a'` (truncated at null byte)
- `'\x00hello'` → `''` (truncated to empty string)
- `'hello\x00'` → `'hello'` (trailing null byte removed)

## Why This Is A Bug

1. **Violates documented round-trip property**: The `VariableCoder` base class documentation explicitly states that `coder.decode(coder.encode(variable)) == variable` should hold.

2. **Silent data corruption**: Null bytes are valid Unicode characters and can appear in legitimate string data. Silently corrupting them violates user expectations.

3. **No warning or error**: Users have no way to know their data is being corrupted.

## Root Cause

In `encode_string_array` (strings.py:100-103):

```python
def encode_string_array(string_array, encoding="utf-8"):
    string_array = np.asarray(string_array)
    encoded = [x.encode(encoding) for x in string_array.ravel()]
    return np.array(encoded, dtype=bytes).reshape(string_array.shape)
```

The issue is on line 103: `np.array(encoded, dtype=bytes)` creates a fixed-width bytes array (e.g., `dtype='S1'` for `b'\x00'`). NumPy's fixed-width bytes/string dtypes use C-style null termination, where `\x00` acts as a string terminator. This causes null bytes to truncate the string.

## Fix

The fix is to explicitly use `dtype=object` instead of `dtype=bytes` to avoid fixed-width bytes arrays:

```diff
diff --git a/xarray/coding/strings.py b/xarray/coding/strings.py
index 1234567..abcdefg 100644
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -100,7 +100,7 @@ def decode_bytes_array(bytes_array, encoding="utf-8"):
 def encode_string_array(string_array, encoding="utf-8"):
     string_array = np.asarray(string_array)
     encoded = [x.encode(encoding) for x in string_array.ravel()]
-    return np.array(encoded, dtype=bytes).reshape(string_array.shape)
+    return np.array(encoded, dtype=object).reshape(string_array.shape)
```

This ensures that the byte strings are stored in an object array without null-termination semantics, preserving all bytes including nulls.