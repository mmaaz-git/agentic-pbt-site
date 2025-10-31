#!/usr/bin/env python3
"""
Minimal reproduction case for scipy.io.matlab digit-prefixed key bug
"""
from io import BytesIO
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

# Test case 1: Single digit key
print("=== Test 1: Single digit key '0' ===")
bio = BytesIO()
data = {'0': np.array([1, 2, 3])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio, data)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio.seek(0)
loaded = loadmat(bio)
print(f"'0' in loaded: {'0' in loaded}")
if '0' in loaded:
    print(f"Value of '0': {loaded['0']}")
print()

# Test case 2: Digit-prefixed key
print("=== Test 2: Digit-prefixed key '1test' ===")
bio2 = BytesIO()
data2 = {'1test': np.array([4, 5, 6])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio2, data2)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio2.seek(0)
loaded2 = loadmat(bio2)
print(f"'1test' in loaded: {'1test' in loaded2}")
if '1test' in loaded2:
    print(f"Value of '1test': {loaded2['1test']}")
print()

# Test case 3: For comparison - underscore-prefixed key (should be ignored)
print("=== Test 3: Underscore-prefixed key '_hidden' (for comparison) ===")
bio3 = BytesIO()
data3 = {'_hidden': np.array([7, 8, 9])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio3, data3)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio3.seek(0)
loaded3 = loadmat(bio3)
print(f"'_hidden' in loaded: {'_hidden' in loaded3}")
if '_hidden' in loaded3:
    print(f"Value of '_hidden': {loaded3['_hidden']}")
print()

# Test case 4: Struct field with digit prefix (should be ignored per existing behavior)
print("=== Test 4: Struct field with digit prefix '0field' (for comparison) ===")
bio4 = BytesIO()
data4 = {'mystruct': {'0field': np.array([10, 11, 12]), 'valid_field': np.array([13, 14, 15])}}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio4, data4)
    print(f"Warnings issued: {len(w)}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

bio4.seek(0)
loaded4 = loadmat(bio4)
if 'mystruct' in loaded4:
    struct_data = loaded4['mystruct']
    print(f"Struct fields: {struct_data.dtype.names}")
    if struct_data.dtype.names and '0field' in struct_data.dtype.names:
        print(f"  '0field' was saved (unexpected)")
    else:
        print(f"  '0field' was not saved (expected)")