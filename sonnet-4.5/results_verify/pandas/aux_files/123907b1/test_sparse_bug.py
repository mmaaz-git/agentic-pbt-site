#!/usr/bin/env python3
"""Test to reproduce the SparseArray boolean comparison bug."""

from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray
import traceback
import pandas as pd
import numpy as np

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print()

# First, let's test the hypothesis property test
@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    fill_value = draw(st.booleans())
    kind = draw(st.sampled_from(['integer', 'block']))
    return SparseArray(values, fill_value=fill_value, kind=kind)

@given(sparse_arrays(), sparse_arrays())
@settings(max_examples=10)  # Reduced for faster testing
def test_equality_symmetric(arr1, arr2):
    """If a.equals(b), then b.equals(a)"""
    try:
        # Test equality comparison
        result = arr1 == arr2
        print(f"✓ Comparison succeeded for arrays of length {len(arr1)} and {len(arr2)}")
    except AttributeError as e:
        print(f"✗ AttributeError caught: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise

    # Also test the .equals() method
    if arr1.equals(arr2):
        assert arr2.equals(arr1), "Equality not symmetric"

print("Running hypothesis test...")
print("-" * 50)
try:
    test_equality_symmetric()
    print("Hypothesis test completed without errors")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")
    traceback.print_exc()

print()
print("=" * 50)
print("Now testing the minimal reproduction example...")
print("-" * 50)

# Minimal reproduction
try:
    arr1 = SparseArray([False, False, True], fill_value=True)
    arr2 = SparseArray([False, True, True], fill_value=True)

    print(f"arr1: {arr1}")
    print(f"arr1 dtype: {arr1.dtype}")
    print(f"arr1 fill_value: {arr1.fill_value}")
    print()
    print(f"arr2: {arr2}")
    print(f"arr2 dtype: {arr2.dtype}")
    print(f"arr2 fill_value: {arr2.fill_value}")
    print()

    print("Attempting comparison: arr1 == arr2")
    result = arr1 == arr2
    print(f"Success! Result: {result}")
    print(f"Result type: {type(result)}")
except AttributeError as e:
    print(f"✗ AttributeError as reported in bug: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"✗ Different error than reported: {e}")
    traceback.print_exc()

print()
print("=" * 50)
print("Testing other comparison operators...")
print("-" * 50)

operators = [
    ("==", "eq"),
    ("!=", "ne"),
    ("<", "lt"),
    ("<=", "le"),
    (">", "gt"),
    (">=", "ge"),
]

for op_symbol, op_name in operators:
    try:
        arr1 = SparseArray([False, False, True], fill_value=True)
        arr2 = SparseArray([False, True, True], fill_value=True)

        if op_symbol == "==":
            result = arr1 == arr2
        elif op_symbol == "!=":
            result = arr1 != arr2
        elif op_symbol == "<":
            result = arr1 < arr2
        elif op_symbol == "<=":
            result = arr1 <= arr2
        elif op_symbol == ">":
            result = arr1 > arr2
        elif op_symbol == ">=":
            result = arr1 >= arr2

        print(f"✓ {op_symbol} ({op_name}) succeeded")
    except AttributeError as e:
        print(f"✗ {op_symbol} ({op_name}) failed with AttributeError: {e}")
    except Exception as e:
        print(f"✗ {op_symbol} ({op_name}) failed with: {e}")

print()
print("=" * 50)
print("Testing logical operations (for comparison)...")
print("-" * 50)

# Test logical operations which supposedly work
logical_ops = [
    ("&", "and"),
    ("|", "or"),
    ("^", "xor"),
]

for op_symbol, op_name in logical_ops:
    try:
        arr1 = SparseArray([False, False, True], fill_value=False)
        arr2 = SparseArray([False, True, True], fill_value=False)

        if op_symbol == "&":
            result = arr1 & arr2
        elif op_symbol == "|":
            result = arr1 | arr2
        elif op_symbol == "^":
            result = arr1 ^ arr2

        print(f"✓ {op_symbol} ({op_name}) succeeded: {result.to_dense()}")
    except Exception as e:
        print(f"✗ {op_symbol} ({op_name}) failed: {e}")