# Bug Report: xarray.coding CFMaskCoder Fill Value Round-Trip Violation

**Target**: `xarray.coding.variables.CFMaskCoder`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CFMaskCoder` violates the fundamental round-trip property `decode(encode(var)) == var` when data contains values equal to the fill value. Valid data values matching the fill value are incorrectly treated as missing data and replaced with NaN during decoding.

## Property-Based Test

```python
@given(arrays(dtype=np.float32, shape=st.tuples(st.integers(5, 20))),
       st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_mask_coder_with_fill_value_roundtrip(self, data, fill_value):
    assume(not np.any(np.isnan(data)))
    assume(not np.any(np.isinf(data)))
    assume(not np.isnan(fill_value))

    original_var = Variable(('x',), data.copy(), encoding={'_FillValue': fill_value})
    coder = CFMaskCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)
```

**Failing input**: `data=array([0., 0., 0., 0., 0.], dtype=float32), fill_value=0.0`

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.variables import CFMaskCoder
from xarray.core.variable import Variable

data = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
fill_value = 0.0

original_var = Variable(('x',), data.copy(), encoding={'_FillValue': fill_value})
coder = CFMaskCoder()

encoded_var = coder.encode(original_var)
decoded_var = coder.decode(encoded_var)

print("Original:", original_var.data)
print("Decoded:", decoded_var.data)
```

Output:
```
Original: [0. 0. 0. 0. 0.]
Decoded: [nan nan nan nan nan]
```

## Why This Is A Bug

The `VariableCoder` base class documentation explicitly states:

> Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`.

When data contains valid values that happen to match the fill value (e.g., actual zeros when `_FillValue=0.0`), the decoder incorrectly treats all such values as missing data and replaces them with NaN. This violates the fundamental round-trip property and corrupts valid data.

This is particularly problematic because:
1. Zero is a common and valid data value in scientific datasets
2. Zero is often used as a fill value in netCDF files
3. The corruption is silent - no warning is issued

## Fix

The issue is in the interaction between encoding and decoding. During encoding, when data values equal the fill value, they should either:
1. Be preserved as-is if they represent valid data
2. Already be NaN/masked if they represent missing data

The decoder should only convert fill values to NaN if they were explicitly marked as missing during encoding, not all occurrences of the fill value in the data.

A proper fix would require tracking which values are truly missing vs. which are valid data that happen to equal the fill value. This might involve:
- Using a separate mask array during encoding/decoding
- Checking if the original data contains NaN/masked values before encoding
- Only applying the fill value mask to values that were originally NaN/masked