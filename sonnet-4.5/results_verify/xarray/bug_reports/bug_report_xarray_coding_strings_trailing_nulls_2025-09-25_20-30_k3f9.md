# Bug Report: xarray.coding.strings Trailing Null Byte Loss

**Target**: `xarray.coding.strings.encode_string_array` / `decode_bytes_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `encode_string_array` and `decode_bytes_array` functions violate their round-trip property when strings contain trailing null bytes (`\x00`). Trailing null bytes are silently stripped during encoding, causing data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=100))
@settings(max_examples=1000)
def test_string_encoding_round_trip(string_list):
    """
    Property: decode_bytes_array(encode_string_array(arr)) == arr
    String encoding should round-trip for UTF-8 encoding.
    """
    arr = np.array(string_list, dtype=object)
    encoded = encode_string_array(arr)
    decoded = decode_bytes_array(encoded)
    assert np.array_equal(decoded, arr)
```

**Failing input**: `string_list=['\x00']` (and any string with trailing null bytes like `'hello\x00'`)

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

original = np.array(['\x00'], dtype=object)
print(f"Original: {original[0]!r}, length={len(original[0])}")

encoded = encode_string_array(original)
print(f"Encoded: {encoded[0]!r}, length={len(encoded[0])}")

decoded = decode_bytes_array(encoded)
print(f"Decoded: {decoded[0]!r}, length={len(decoded[0])}")

assert np.array_equal(decoded, original)
```

Output:
```
Original: '\x00', length=1
Encoded: b'', length=0
Decoded: '', length=0
AssertionError
```

Additional failing examples:
- `'hello\x00'` → encodes to `b'hello'` (null byte stripped)
- `'test\x00\x00'` → encodes to `b'test'` (both null bytes stripped)
- `'a\x00b\x00'` → encodes to `b'a\x00b'` (only trailing null stripped)

## Why This Is A Bug

1. **Violates documented round-trip property**: The xarray codebase's `VariableCoder` base class explicitly states that coders should satisfy `coder.decode(coder.encode(variable)) == variable`.

2. **Silent data corruption**: Trailing null bytes are valid in Unicode strings and can have semantic meaning. Silently stripping them corrupts user data without warning.

3. **Inconsistent behavior**: Null bytes in the middle of strings are preserved, but trailing ones are stripped. This is surprising and inconsistent.

4. **Root cause**: The bug occurs in `encode_string_array` (strings.py:100-103) because numpy's fixed-width byte string dtype (`'S'`) automatically strips trailing null bytes when determining array dtype. This is a fundamental limitation of numpy's `'S'` dtype.

## Fix

The current implementation:
```python
def encode_string_array(string_array, encoding="utf-8"):
    string_array = np.asarray(string_array)
    encoded = [x.encode(encoding) for x in string_array.ravel()]
    return np.array(encoded, dtype=bytes).reshape(string_array.shape)
```

Cannot be fixed by changing the dtype parameter because numpy's `'S'` dtype fundamentally strips trailing nulls. The fix requires using object dtype to preserve bytes exactly:

```diff
def encode_string_array(string_array, encoding="utf-8"):
    string_array = np.asarray(string_array)
    encoded = [x.encode(encoding) for x in string_array.ravel()]
-   return np.array(encoded, dtype=bytes).reshape(string_array.shape)
+   return np.array(encoded, dtype=object).reshape(string_array.shape)
```

This preserves the exact bytes including trailing nulls, at the cost of using object arrays instead of fixed-width byte arrays. Since `decode_bytes_array` already handles object dtype arrays correctly (it iterates over elements with `.ravel()`), this fix maintains compatibility while fixing the bug.