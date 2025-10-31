#!/usr/bin/env python3
"""Test the reported scipy.fftpack.dct bug with single-element arrays"""

import numpy as np
import scipy.fftpack as fftpack

print("Testing scipy.fftpack.dct with single-element array...")
print("=" * 60)

# Test the exact case from the bug report
x = np.array([1.])
print(f"Input array: {x}")
print(f"Input shape: {x.shape}")
print()

# Test DCT Type 1 (expected to fail according to bug report)
print("Testing DCT Type 1:")
try:
    result = fftpack.dct(x, type=1)
    print(f"  Success! Result: {result}")
except Exception as e:
    print(f"  Failed with error: {type(e).__name__}: {e}")

print()

# Test other DCT types for comparison
for dct_type in [2, 3, 4]:
    print(f"Testing DCT Type {dct_type}:")
    try:
        result = fftpack.dct(x, type=dct_type)
        print(f"  Success! Result: {result}")
    except Exception as e:
        print(f"  Failed with error: {type(e).__name__}: {e}")

print()
print("=" * 60)

# Also test with normalization
print("Testing with norm='ortho':")
for dct_type in [1, 2, 3, 4]:
    print(f"  Type {dct_type}:", end=" ")
    try:
        result = fftpack.dct(x, type=dct_type, norm='ortho')
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Failed - {type(e).__name__}: {e}")

print()
print("=" * 60)

# Test with 2-element array for comparison
x2 = np.array([1., 2.])
print(f"\nTesting with 2-element array: {x2}")
for dct_type in [1, 2, 3, 4]:
    print(f"  Type {dct_type}:", end=" ")
    try:
        result = fftpack.dct(x2, type=dct_type)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Failed - {type(e).__name__}: {e}")