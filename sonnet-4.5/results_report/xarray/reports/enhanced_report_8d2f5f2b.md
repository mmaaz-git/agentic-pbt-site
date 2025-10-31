# Bug Report: xarray.coding.strings Null Character Data Loss

**Target**: `xarray.coding.strings.encode_string_array` and `decode_bytes_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The string encoding/decoding functions in xarray lose null characters (`\x00`) when they appear at the beginning or end of strings, violating the documented round-trip property that `decode(encode(x)) == x`.

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

if __name__ == "__main__":
    test_string_encode_decode_round_trip()
```

<details>

<summary>
**Failing input**: `['\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 14, in <module>
    test_string_encode_decode_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 6, in test_string_encode_decode_round_trip
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 11, in test_string_encode_decode_round_trip
    assert np.array_equal(decoded, arr), f"Round-trip failed: {arr} != {decoded}"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^
AssertionError: Round-trip failed: ['\x00'] != ['']
Falsifying example: test_string_encode_decode_round_trip(
    strings=['\x00'],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1708
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

# Test with a single null character
arr = np.array(['\x00'], dtype=object)
encoded = encode_string_array(arr, encoding="utf-8")
decoded = decode_bytes_array(encoded, encoding="utf-8")

print(f"Original: {arr[0]!r} (length {len(arr[0])})")
print(f"Decoded:  {decoded[0]!r} (length {len(decoded[0])})")
print(f"Equal: {arr[0] == decoded[0]}")

# Test with multiple null characters
arr2 = np.array(['\x00\x00\x00'], dtype=object)
encoded2 = encode_string_array(arr2, encoding="utf-8")
decoded2 = decode_bytes_array(encoded2, encoding="utf-8")
print(f"\nOriginal: {arr2[0]!r} (length {len(arr2[0])})")
print(f"Decoded:  {decoded2[0]!r} (length {len(decoded2[0])})")

# Test with trailing null
arr3 = np.array(['test\x00'], dtype=object)
encoded3 = encode_string_array(arr3, encoding="utf-8")
decoded3 = decode_bytes_array(encoded3, encoding="utf-8")
print(f"\nOriginal: {arr3[0]!r} (length {len(arr3[0])})")
print(f"Decoded:  {decoded3[0]!r} (length {len(decoded3[0])})")

# Test with embedded null (which might work)
arr4 = np.array(['hello\x00world'], dtype=object)
encoded4 = encode_string_array(arr4, encoding="utf-8")
decoded4 = decode_bytes_array(encoded4, encoding="utf-8")
print(f"\nOriginal: {arr4[0]!r} (length {len(arr4[0])})")
print(f"Decoded:  {decoded4[0]!r} (length {len(decoded4[0])})")

# Final assertion to demonstrate failure
assert arr[0] == decoded[0], f"Round-trip failed: {arr[0]!r} != {decoded[0]!r}"
```

<details>

<summary>
Output showing data loss
</summary>
```
Original: '\x00' (length 1)
Decoded:  '' (length 0)
Equal: False

Original: '\x00\x00\x00' (length 3)
Decoded:  '' (length 0)

Original: 'test\x00' (length 5)
Decoded:  'test' (length 4)

Original: 'hello\x00world' (length 11)
Decoded:  'hello\x00world' (length 11)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/repo.py", line 35, in <module>
    assert arr[0] == decoded[0], f"Round-trip failed: {arr[0]!r} != {decoded[0]!r}"
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: '\x00' != ''
```
</details>

## Why This Is A Bug

This violates the documented contract in the `VariableCoder` base class (line 29-30 in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/coding/common.py`):

> "Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`"

The bug causes silent data corruption where null bytes are lost without any warning or error. The root cause is that `encode_string_array` uses `np.array(encoded, dtype=bytes)` on line 103, which creates a fixed-width byte string dtype (e.g., `|S1`, `|S5`). NumPy treats null bytes as C-style string terminators in these dtypes, truncating the data:

- Leading null bytes (`\x00`) become empty strings
- Trailing null bytes (`test\x00`) are truncated
- Embedded null bytes (`hello\x00world`) are preserved because NumPy reads up to the dtype's fixed width

## Relevant Context

The `EncodedStringCoder` class inherits from `VariableCoder` and uses these functions internally. This affects users who:
- Store binary data or control characters in xarray string variables
- Serialize/deserialize xarray datasets containing null bytes
- Expect data integrity when using xarray's CF-compliant encoding

While null bytes in strings are relatively uncommon in typical scientific computing workflows, the silent data loss without warning makes this a data integrity issue that should be addressed.

## Proposed Fix

Replace the problematic line that infers a fixed-width dtype with explicit object dtype to preserve null bytes:

```diff
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -100,7 +100,11 @@ def encode_string_array(string_array, encoding="utf-8"):
 def encode_string_array(string_array, encoding="utf-8"):
     string_array = np.asarray(string_array)
     encoded = [x.encode(encoding) for x in string_array.ravel()]
-    return np.array(encoded, dtype=bytes).reshape(string_array.shape)
+    # Use object dtype to avoid null byte truncation in fixed-width byte strings
+    # NumPy's fixed-width byte string dtypes (|S1, |S5, etc.) treat null bytes
+    # as C-style string terminators, causing data loss
+    result = np.empty(len(encoded), dtype=object)
+    result[:] = encoded
+    return result.reshape(string_array.shape)
```