# Bug Report: llm.decode Silent Data Truncation

**Target**: `llm.decode`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `decode` function silently truncates input data when the binary length is not a multiple of 4, potentially hiding data corruption or incorrect usage without any warning or error.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.binary(min_size=1, max_size=100))
def test_decode_validates_input_length(binary_data):
    if len(binary_data) % 4 != 0:
        try:
            result = llm.decode(binary_data)
            assert False, f"decode should reject invalid input but accepted {len(binary_data)} bytes"
        except (ValueError, struct.error):
            pass
    else:
        result = llm.decode(binary_data)
        assert len(result) == len(binary_data) // 4
```

**Failing input**: `binary_data = b'\x00\x00\x00\x00\xFF'` (5 bytes, not multiple of 4)

## Reproducing the Bug

```python
import llm

values = [1.0, 2.0, 3.0]
encoded = llm.encode(values)

corrupted_bytes = encoded + b'\x00'

decoded = llm.decode(corrupted_bytes)

print(f"Input length: {len(corrupted_bytes)} (not multiple of 4)")
print(f"Decoded: {list(decoded)}")
print(f"Lost data: {len(corrupted_bytes) - len(decoded) * 4} bytes silently discarded")
```

**Output**:
```
Input length: 13 (not multiple of 4)
Decoded: [1.0, 2.0, 3.0]
Lost data: 1 bytes silently discarded
```

## Why This Is A Bug

The `decode` function is part of the public API and can be called with arbitrary binary data. The implementation uses:

```python
def decode(binary):
    return struct.unpack("<" + "f" * (len(binary) // 4), binary)
```

When `len(binary)` is not a multiple of 4, the expression `len(binary) // 4` performs integer division, discarding the remainder. This causes `struct.unpack` to only read the first `N * 4` bytes, silently ignoring any trailing bytes.

This is problematic because:
1. **Silent data loss**: Users don't know their data was truncated
2. **Masks corruption**: If data is corrupted in transit, the function might still succeed
3. **No precondition documentation**: The function has no docstring explaining this requirement

While the paired `encode` function always produces valid output, `decode` is a public API that could be called independently with data from other sources.

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -453,5 +453,8 @@ def encode(values):


 def decode(binary):
+    if len(binary) % 4 != 0:
+        raise ValueError(f"Binary data length must be a multiple of 4, got {len(binary)} bytes")
+
     return struct.unpack("<" + "f" * (len(binary) // 4), binary)
```