# Bug Report: xarray.coding CFScaleOffsetCoder Division by Zero

**Target**: `xarray.coding.variables.CFScaleOffsetCoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`CFScaleOffsetCoder` crashes and violates the round-trip property when `scale_factor=0.0`. The encoder divides by zero during encoding, producing infinities, which then become NaNs during decoding. This corrupts all data and violates the fundamental property `decode(encode(var)) == var`.

## Property-Based Test

```python
@given(arrays(dtype=np.float32, shape=st.tuples(st.integers(5, 20))))
@settings(max_examples=50)
def test_scale_offset_coder_zero_scale(self, data):
    assume(not np.any(np.isnan(data)))
    assume(not np.any(np.isinf(data)))

    scale_factor = 0.0
    add_offset = 10.0

    original_var = Variable(('x',), data.copy(),
                          encoding={'scale_factor': scale_factor, 'add_offset': add_offset})
    coder = CFScaleOffsetCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)
```

**Failing input**: `data=array([1., 2., 3., 4., 5.], dtype=float32), scale_factor=0.0`

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.variables import CFScaleOffsetCoder
from xarray.core.variable import Variable

data = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
scale_factor = 0.0
add_offset = 10.0

original_var = Variable(('x',), data.copy(),
                      encoding={'scale_factor': scale_factor, 'add_offset': add_offset})
coder = CFScaleOffsetCoder()

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    encoded_var = coder.encode(original_var)

decoded_var = coder.decode(encoded_var)

print("Original:", original_var.data)
print("Encoded:", encoded_var.data)
print("Decoded:", decoded_var.data)
```

Output:
```
Original: [1. 2. 3. 4. 5.]
Encoded: [-inf -inf -inf -inf -inf]
Decoded: [nan nan nan nan nan]
```

## Why This Is A Bug

The `VariableCoder` base class documentation explicitly states:

> Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`.

When `scale_factor=0.0`, the encoding step in `variables.py:518` performs:
```python
data /= pop_to(encoding, attrs, "scale_factor", name=name)
```

This divides by zero, producing infinities. During decoding, multiplying infinities by zero produces NaN, completely corrupting the data.

While `scale_factor=0.0` might be considered invalid input, the coder should either:
1. Validate the input and raise a clear error message
2. Handle it gracefully to maintain the round-trip property

Currently it silently corrupts data with only a RuntimeWarning.

## Fix

Add validation in `CFScaleOffsetCoder.encode()` to check for zero or near-zero scale factors:

```diff
--- a/xarray/coding/variables.py
+++ b/xarray/coding/variables.py
@@ -515,6 +515,10 @@ class CFScaleOffsetCoder(VariableCoder):
         if "add_offset" in encoding or "scale_factor" in encoding:
             data = data.astype(dtype=float, copy=True)
         if "scale_factor" in encoding:
+            scale_factor = encoding.get("scale_factor")
+            if scale_factor == 0.0 or np.isclose(scale_factor, 0.0):
+                raise ValueError(
+                    f"scale_factor must be non-zero, got {scale_factor}")
             data /= pop_to(encoding, attrs, "scale_factor", name=name)
         if "add_offset" in encoding:
             data -= pop_to(encoding, attrs, "add_offset", name=name)
```