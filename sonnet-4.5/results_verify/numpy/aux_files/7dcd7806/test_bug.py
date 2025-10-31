#!/usr/bin/env python3
"""Test script to reproduce the numpy.rec.fromrecords empty list bug"""

import numpy as np
import numpy.rec
from hypothesis import given, strategies as st
import traceback

print("Testing numpy.rec.fromrecords and numpy.rec.fromarrays with empty lists")
print("=" * 70)

# First test: The hypothesis test from the bug report
print("\n1. Running Hypothesis test:")
print("-" * 40)

@given(st.lists(st.tuples(st.integers(), st.text(max_size=10)), min_size=0, max_size=20))
def test_fromrecords_empty_handling(records):
    rec = numpy.rec.fromrecords(records, names='a,b')
    assert len(rec) == len(records)

try:
    test_fromrecords_empty_handling()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

# Second test: Direct reproduction with fromrecords
print("\n2. Testing numpy.rec.fromrecords with empty list:")
print("-" * 40)

try:
    result = numpy.rec.fromrecords([], names='a,b')
    print(f"Success! Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Third test: Direct reproduction with fromarrays
print("\n3. Testing numpy.rec.fromarrays with empty list:")
print("-" * 40)

try:
    result = numpy.rec.fromarrays([], names='a,b')
    print(f"Success! Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Fourth test: Regular numpy array with empty list (for comparison)
print("\n4. Testing regular numpy.array with empty list (for comparison):")
print("-" * 40)

try:
    result = np.array([])
    print(f"Success! Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Fifth test: Test with non-empty lists to verify normal operation
print("\n5. Testing with non-empty lists (control test):")
print("-" * 40)

try:
    records = [(1, 'a'), (2, 'b'), (3, 'c')]
    result = numpy.rec.fromrecords(records, names='x,y')
    print(f"fromrecords with data - Success!")
    print(f"Result: {result}")
    print(f"Length: {len(result)}")

    arrays = [[1, 2, 3], ['a', 'b', 'c']]
    result2 = numpy.rec.fromarrays(arrays, names='x,y')
    print(f"\nfromarrays with data - Success!")
    print(f"Result: {result2}")
    print(f"Length: {len(result2)}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Sixth test: Check other ways to create empty structured arrays
print("\n6. Testing alternative ways to create empty structured arrays:")
print("-" * 40)

try:
    # Using numpy.zeros with structured dtype
    dtype = np.dtype([('a', int), ('b', 'U10')])
    result = np.zeros(0, dtype=dtype)
    print(f"np.zeros with structured dtype - Success!")
    print(f"Result: {result}")
    print(f"Shape: {result.shape}")
    print(f"Dtype: {result.dtype}")

    # Convert to recarray
    rec = result.view(np.recarray)
    print(f"\nConverted to recarray: {rec}")
    print(f"Type: {type(rec)}")

except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("Testing complete")