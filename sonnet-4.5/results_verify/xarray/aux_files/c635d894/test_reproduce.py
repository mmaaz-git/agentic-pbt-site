import numpy as np
import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.coding.variables import CFScaleOffsetCoder
from xarray.core.variable import Variable

# Test with the reported failing input
data = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
scale_factor = 0.0
add_offset = 10.0

original_var = Variable(('x',), data.copy(),
                      encoding={'scale_factor': scale_factor, 'add_offset': add_offset})
coder = CFScaleOffsetCoder()

import warnings
print("Testing CFScaleOffsetCoder with scale_factor=0.0")
print("=" * 50)
print(f"Original data: {original_var.data}")
print(f"scale_factor: {scale_factor}, add_offset: {add_offset}")
print()

# Encode step
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    encoded_var = coder.encode(original_var)
    if w:
        print("Warnings during encoding:")
        for warning in w:
            print(f"  - {warning.message}")
    else:
        print("No warnings during encoding")

print(f"Encoded data: {encoded_var.data}")
print()

# Decode step
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    decoded_var = coder.decode(encoded_var)
    if w:
        print("Warnings during decoding:")
        for warning in w:
            print(f"  - {warning.message}")
    else:
        print("No warnings during decoding")

print(f"Decoded data: {decoded_var.data}")
print()

# Check if round-trip property is satisfied
print("Round-trip property check:")
if np.array_equal(original_var.data, decoded_var.data, equal_nan=False):
    print("✓ decode(encode(var)) == var")
else:
    print("✗ decode(encode(var)) != var")
    print(f"  Expected: {original_var.data}")
    print(f"  Got: {decoded_var.data}")

# Also check if all decoded values are NaN
if np.all(np.isnan(decoded_var.data)):
    print("✗ All decoded values are NaN - data is completely corrupted!")