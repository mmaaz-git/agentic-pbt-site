#!/usr/bin/env python3
"""Minimal reproduction of numpy.ma clump_masked/clump_unmasked bug with empty arrays."""

import numpy as np
import numpy.ma as ma

print("Testing numpy.ma clump functions with empty arrays")
print("=" * 60)

# Create an empty masked array
empty_arr = ma.array([], dtype=int, mask=[])
print(f"Input array: {empty_arr}")
print(f"Array shape: {empty_arr.shape}")
print(f"Array mask: {ma.getmask(empty_arr)}")
print()

# Test clump_masked
print("Testing ma.clump_masked() with empty array:")
print("-" * 40)
try:
    result = ma.clump_masked(empty_arr)
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()

print()

# Test clump_unmasked
print("Testing ma.clump_unmasked() with empty array:")
print("-" * 40)
try:
    result = ma.clump_unmasked(empty_arr)
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()