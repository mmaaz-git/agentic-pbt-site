# Bug Report: xarray.coding.variables.CFMaskCoder Round-Trip Data Corruption

**Target**: `xarray.coding.variables.CFMaskCoder`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CFMaskCoder` violates the documented round-trip property when data contains values equal to the `_FillValue`. Valid non-NaN data is silently corrupted to NaN during encode/decode, causing data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from xarray.core.variable import Variable
from xarray.coding.variables import CFMaskCoder

@given(
    st.lists(st.floats(allow_nan=False, min_value=-1e6, max_value=1e6),
             min_size=4, max_size=100),
    st.floats(allow_nan=False, min_value=-1e6, max_value=1e6)
)
def test_cfmask_coder_round_trip(floats, fill_value):
    arr = np.array(floats[:4], dtype=np.float32).reshape((2, 2))
    var = Variable(('x', 'y'), arr, encoding={'_FillValue': fill_value})

    coder = CFMaskCoder()
    encoded = coder.encode(var)
    decoded = coder.decode(encoded)

    # Round-trip should preserve non-NaN values
    nan_mask_orig = np.isnan(var.data)
    nan_mask_decoded = np.isnan(decoded.data)
    assert np.array_equal(nan_mask_orig, nan_mask_decoded)
```

**Failing input**: `floats=[0.0, 0.0, 0.0, 0.0], fill_value=0.0`

## Reproducing the Bug

```python
import numpy as np
from xarray.core.variable import Variable
from xarray.coding.variables import CFMaskCoder

data = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32)
var = Variable(('x',), data, encoding={'_FillValue': 0.0})

print(f"Original data: {data}")
print(f"NaNs in original: {np.isnan(data)}")

coder = CFMaskCoder()
encoded = coder.encode(var)
decoded = coder.decode(encoded)

print(f"Decoded data: {decoded.data}")
print(f"NaNs in decoded: {np.isnan(decoded.data)}")

assert not np.array_equal(np.isnan(data), np.isnan(decoded.data))
```

Output:
```
Original data: [0. 1. 2. 0.]
NaNs in original: [False False False False]
Decoded data: [nan  1.  2. nan]
NaNs in decoded: [ True False False  True]
```

## Why This Is A Bug

1. **Violates documented contract**: The `VariableCoder` docstring (xarray/coding/common.py:30) explicitly states: "Subclasses should implement encode() and decode(), which should satisfy the identity `coder.decode(coder.encode(variable)) == variable`."

2. **Data corruption**: Valid non-NaN data values that happen to equal the fill value are silently converted to NaN, causing irreversible data loss.

3. **Realistic scenario**: This commonly occurs when:
   - Using 0.0 as fill value with data containing legitimate zeros
   - Using -999 as fill value with data containing that sentinel value
   - Any case where fill value appears in actual data

## Fix

The bug occurs because `CFMaskCoder.decode()` treats all occurrences of the fill value as missing data, regardless of whether they were originally NaN. The encode method should only replace actual NaN values with the fill value, and the decode method should only replace fill values that were originally NaN.

One potential fix: track which values were originally NaN during encoding so they can be properly restored during decoding. However, this may not be feasible within the CF conventions framework.

Alternatively, the documentation should be updated to clarify that the round-trip property only holds when the data does not contain values equal to the fill value.