#!/usr/bin/env python3
"""Test the bug report for _handle_truncated_float_vec"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.io.sas.sas_xport import _handle_truncated_float_vec

# First, run the hypothesis test
print("Running hypothesis test...")

@given(nbytes=st.integers(min_value=9, max_value=100))
@settings(max_examples=10)  # Reduced for quicker testing
def test_handle_truncated_float_vec_invalid_nbytes(nbytes):
    vec = np.array([b'TEST'], dtype='S4')

    try:
        result = _handle_truncated_float_vec(vec, nbytes)
        assert False, f"Should raise error for nbytes={nbytes}"
    except Exception as e:
        print(f"  nbytes={nbytes}: Got {type(e).__name__}: {e}")
        pass

test_handle_truncated_float_vec_invalid_nbytes()

print("\nReproducing the specific bug with nbytes=9...")

vec = np.array([b'TEST'], dtype='S4')

try:
    result = _handle_truncated_float_vec(vec, 9)
    print(f"ERROR: Should have raised error, got: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

print("\nTesting edge cases...")

# Test nbytes=0
print("Testing nbytes=0:")
try:
    result = _handle_truncated_float_vec(vec, 0)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Test nbytes=1
print("Testing nbytes=1:")
try:
    result = _handle_truncated_float_vec(vec, 1)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Test nbytes=-1
print("Testing nbytes=-1:")
try:
    result = _handle_truncated_float_vec(vec, -1)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Test valid range
print("\nTesting valid range (2-8):")
for nbytes in range(2, 9):
    try:
        result = _handle_truncated_float_vec(vec, nbytes)
        print(f"  nbytes={nbytes}: Success, shape={result.shape if hasattr(result, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"  nbytes={nbytes}: {type(e).__name__}: {e}")