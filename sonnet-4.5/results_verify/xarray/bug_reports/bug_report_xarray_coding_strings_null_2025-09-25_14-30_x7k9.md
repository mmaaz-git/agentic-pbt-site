# Bug Report: xarray.coding.strings Null Character Data Loss

**Target**: `xarray.coding.strings.encode_string_array` and `decode_bytes_array`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

String encoding/decoding in xarray loses null characters (`\x00`), violating the documented round-trip property that `decode(encode(x)) == x`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
@settings(max_examples=1000)
def test_string_encode_decode_round_trip(strings):
    arr = np.array(strings, dtype=object)
    encoded = encode_string_array(arr, encoding="utf-8")
    decoded = decode_bytes_array(encoded, encoding="utf-8")
    assert np.array_equal(decoded, arr), f"Round-trip failed: {arr} != {decoded}"
```

**Failing input**: `['\x00']` (string containing a single null character)

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

arr = np.array(['\x00'], dtype=object)
encoded = encode_string_array(arr, encoding="utf-8")
decoded = decode_bytes_array(encoded, encoding="utf-8")

print(f"Original: {arr[0]!r} (length {len(arr[0])})")
print(f"Decoded:  {decoded[0]!r} (length {len(decoded[0])})")
assert arr[0] == decoded[0]
```

Output:
```
Original: '\x00' (length 1)
Decoded:  '' (length 0)
AssertionError
```

Additional failing cases:
- `'\x00\x00\x00'` → `''` (all null characters lost)
- `'test\x00'` → `'test'` (trailing null character lost)
- `'hello\x00world'` → `'hello\x00world'` (✓ works when null is embedded)

## Why This Is A Bug

1. **Violates documented contract**: The `VariableCoder` base class (line 30 in `common.py`) explicitly states: "Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`"

2. **Silent data corruption**: Users storing strings with null characters will lose data without any error or warning when using xarray's encoding/decoding.

3. **Root cause**: The `encode_string_array` function uses `np.array(encoded, dtype=bytes)`, which creates a fixed-width byte string dtype (e.g., `|S1`). NumPy treats null bytes as C-style string terminators in these dtypes, truncating the data.

## Fix

The bug occurs because NumPy's fixed-width byte string dtypes (`|S1`, `|S5`, etc.) treat null bytes as string terminators. The fix is to avoid inferring the dtype and instead use object dtype for the intermediate encoded array:

```diff
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -100,7 +100,9 @@ def encode_string_array(string_array, encoding="utf-8"):
 def encode_string_array(string_array, encoding="utf-8"):
     string_array = np.asarray(string_array)
     encoded = [x.encode(encoding) for x in string_array.ravel()]
-    return np.array(encoded, dtype=bytes).reshape(string_array.shape)
+    # Use object dtype to avoid null byte truncation in fixed-width byte strings
+    result = np.empty(len(encoded), dtype=object)
+    result[:] = encoded
+    return result.reshape(string_array.shape)
```

Alternatively, if fixed-width dtypes are required for compatibility, the function should raise an error when null bytes are present rather than silently corrupting data.