#!/usr/bin/env python3
import numpy as np
import numpy.rec as rec
import traceback

print("Test 1: Reproducing the exact bug report scenario")
try:
    result = rec.array([], dtype=[('x', 'i4')])
    print(f"Success: Created empty recarray: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nTest 2: Empty list without dtype")
try:
    result = rec.array([])
    print(f"Success: Created empty recarray: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 3: Empty tuple with dtype")
try:
    result = rec.array((), dtype=[('x', 'i4')])
    print(f"Success: Created empty recarray: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 4: Comparing with standard numpy array")
try:
    standard_arr = np.array([], dtype=[('x', 'i4')])
    print(f"Standard np.array with empty list works: {standard_arr}")
    print(f"Shape: {standard_arr.shape}, dtype: {standard_arr.dtype}")
except Exception as e:
    print(f"Standard numpy array error: {e}")

print("\nTest 5: Non-empty list with rec.array (should work)")
try:
    result = rec.array([(1,), (2,)], dtype=[('x', 'i4')])
    print(f"Non-empty list works: {result}")
except Exception as e:
    print(f"Error: {e}")