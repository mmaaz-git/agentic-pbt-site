#!/usr/bin/env python3

import numpy.rec
import numpy as np

print("Testing numpy.rec.array with empty list...")
try:
    result = numpy.rec.array([])
    print("Success! Result:")
    print(f"  Type: {type(result)}")
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Let's also test what happens with related functions
print("Testing numpy.rec.fromrecords with empty list...")
try:
    result = numpy.rec.fromrecords([])
    print("Success! Result:")
    print(f"  Type: {type(result)}")
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

print("Testing numpy.rec.fromarrays with empty list...")
try:
    result = numpy.rec.fromarrays([])
    print("Success! Result:")
    print(f"  Type: {type(result)}")
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test with regular numpy array
print("Testing numpy.array with empty list...")
try:
    result = numpy.array([])
    print("Success! Result:")
    print(f"  Type: {type(result)}")
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")