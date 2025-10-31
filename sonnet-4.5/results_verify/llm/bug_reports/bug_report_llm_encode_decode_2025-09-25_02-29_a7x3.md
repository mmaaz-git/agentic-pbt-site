# Bug Report: llm encode/decode Precision Loss

**Target**: `llm.encode()` / `llm.decode()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `encode()` and `decode()` functions lose precision for very small float values, causing them to underflow to zero. This violates the round-trip property that `decode(encode(values)) == values`.

## Property-Based Test

```python
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)))
@settings(max_examples=1000)
def test_encode_decode_round_trip(values):
    encoded = llm.encode(values)
    decoded = llm.decode(encoded)
    assert len(decoded) == len(values)
    for original, recovered in zip(values, decoded):
        assert math.isclose(original, recovered, rel_tol=1e-6)
```

**Failing input**: `[4.484782386619779e-144]`

## Reproducing the Bug

```python
import llm

values = [4.484782386619779e-144]
encoded = llm.encode(values)
decoded = llm.decode(encoded)

print(f"Original: {values[0]}")
print(f"Decoded:  {decoded[0]}")
print(f"Match: {values[0] == decoded[0]}")
```

Output:
```
Original: 4.484782386619779e-144
Decoded:  0.0
Match: False
```

## Why This Is A Bug

The `encode()` function is used for embedding vectors (see `llm/cli.py` line ~3584), and the round-trip property is fundamental: encoding then decoding should preserve the original values. Very small float values (below ~1.4e-45, the minimum positive float32 value) underflow to zero, causing silent data corruption.

While float32 precision may be acceptable for embeddings, the function doesn't document this limitation, and users would reasonably expect values to round-trip correctly.

## Fix

The root cause is that `encode()` uses format `'f'` which is single-precision (32-bit) float. The minimum positive normal float32 value is approximately 1.17549435e-38, and denormalized values go down to ~1.4e-45. Values below this underflow to zero.

Options to fix:

1. **Use double precision**: Change `'f'` to `'d'` (doubles the storage size)
2. **Document the limitation**: Add a note that values outside float32 range will lose precision
3. **Add validation**: Check values are within float32 range and raise an error if not

```diff
diff --git a/llm/__init__.py b/llm/__init__.py
index xxx..xxx 100644
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -450,7 +450,7 @@ def load_keys():
 def encode(values):
-    return struct.pack("<" + "f" * len(values), *values)
+    return struct.pack("<" + "d" * len(values), *values)


 def decode(binary):
-    return struct.unpack("<" + "f" * (len(binary) // 4), binary)
+    return struct.unpack("<" + "d" * (len(binary) // 8), binary)
```