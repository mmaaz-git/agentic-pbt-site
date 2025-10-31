#!/usr/bin/env python3
"""Test script to reproduce the numpy.rec.fromrecords empty list bug."""

import numpy as np
import numpy.rec as rec
import traceback

print("Testing numpy.rec.fromrecords with empty list...")
print("="*60)

# First, let's verify normal behavior works
print("\n1. Testing normal behavior with non-empty list:")
try:
    r = rec.fromrecords([(1, 2), (3, 4)], names='x,y')
    print(f"   Success! Created recarray with {len(r)} records")
    print(f"   r.x = {r.x}")
    print(f"   r.y = {r.y}")
except Exception as e:
    print(f"   ERROR: {e}")
    traceback.print_exc()

# Now test the reported bug with empty list
print("\n2. Testing with empty list (bug reproduction):")
try:
    r = rec.fromrecords([], names='x,y')
    print(f"   Success! Created recarray with {len(r)} records")
    print(f"   r.x = {r.x}")
    print(f"   r.y = {r.y}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test empty list without names
print("\n3. Testing with empty list, no names:")
try:
    r = rec.fromrecords([])
    print(f"   Success! Created recarray with {len(r)} records")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test with dtype specified
print("\n4. Testing with empty list and dtype:")
try:
    dtype = np.dtype([('x', 'i4'), ('y', 'i4')])
    r = rec.fromrecords([], dtype=dtype)
    print(f"   Success! Created recarray with {len(r)} records")
    print(f"   r.x = {r.x}")
    print(f"   r.y = {r.y}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test other empty array creation methods in numpy for comparison
print("\n5. Comparison - other numpy empty array creation methods:")
try:
    # Regular numpy array
    arr = np.array([])
    print(f"   np.array([]) works: shape={arr.shape}, size={arr.size}")

    # Structured array
    dtype = np.dtype([('x', 'i4'), ('y', 'i4')])
    arr = np.array([], dtype=dtype)
    print(f"   np.array([], dtype=structured) works: shape={arr.shape}, size={arr.size}")

    # fromarrays with empty arrays
    r = rec.fromarrays([np.array([]), np.array([])], names='x,y')
    print(f"   rec.fromarrays with empty arrays works: shape={r.shape}, size={r.size}")
except Exception as e:
    print(f"   ERROR: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete.")