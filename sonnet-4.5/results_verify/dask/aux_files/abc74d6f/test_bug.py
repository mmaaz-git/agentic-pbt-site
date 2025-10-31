#!/usr/bin/env python3
"""Test script to reproduce the reported bug in _sort_mixed"""

import numpy as np
import pandas as pd
import sys
import traceback
from dask.dataframe.dask_expr._expr import _sort_mixed

def test_simple_reproduction():
    """Test the exact reproduction case from the bug report"""
    print("Testing simple reproduction case: [(0,)]")
    try:
        values = np.array([(0,)], dtype=object)
        result = _sort_mixed(values)
        print(f"  Result: {result}")
        print(f"  Success! No error occurred")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_multiple_tuples():
    """Test with multiple tuples"""
    print("\nTesting multiple tuples: [(1,), (2,), (0,)]")
    try:
        values = np.array([(1,), (2,), (0,)], dtype=object)
        result = _sort_mixed(values)
        print(f"  Result: {result}")
        print(f"  Success! Result sorted correctly")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

def test_mixed_with_tuples():
    """Test mixed types including tuples"""
    print("\nTesting mixed types with tuples: [1, 'a', (0,), None]")
    try:
        values = np.array([1, 'a', (0,), None], dtype=object)
        result = _sort_mixed(values)
        print(f"  Result: {result}")
        print(f"  Success! Mixed types sorted correctly")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

def test_without_tuples():
    """Test without tuples (should work)"""
    print("\nTesting without tuples: [2, 1, 'b', 'a', None]")
    try:
        values = np.array([2, 1, 'b', 'a', None], dtype=object)
        result = _sort_mixed(values)
        print(f"  Result: {result}")
        print(f"  Expected order: numbers first (1, 2), then strings ('a', 'b'), then None")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

def test_empty_tuples():
    """Test with empty tuple"""
    print("\nTesting with empty tuple: [()]")
    try:
        values = np.array([()], dtype=object)
        result = _sort_mixed(values)
        print(f"  Result: {result}")
        print(f"  Success! Empty tuple handled")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

def test_multi_element_tuples():
    """Test with tuples having multiple elements"""
    print("\nTesting with multi-element tuples: [(1, 2), (0, 3)]")
    try:
        values = np.array([(1, 2), (0, 3)], dtype=object)
        result = _sort_mixed(values)
        print(f"  Result: {result}")
        print(f"  Success! Multi-element tuples handled")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

def investigate_argsort_behavior():
    """Investigate np.argsort behavior on tuples"""
    print("\n=== Investigating np.argsort behavior on tuples ===")

    # Test argsort on single tuple
    values = np.array([(0,)], dtype=object)
    print(f"Values: {values}")
    print(f"Values shape: {values.shape}")
    result = np.argsort(values)
    print(f"np.argsort result: {result}")
    print(f"np.argsort result shape: {result.shape}")
    print(f"np.argsort result ndim: {result.ndim}")

    # Test argsort on multiple tuples
    values2 = np.array([(1,), (0,), (2,)], dtype=object)
    print(f"\nValues2: {values2}")
    print(f"Values2 shape: {values2.shape}")
    result2 = np.argsort(values2)
    print(f"np.argsort result2: {result2}")
    print(f"np.argsort result2 shape: {result2.shape}")
    print(f"np.argsort result2 ndim: {result2.ndim}")

    # Test argsort on strings for comparison
    str_values = np.array(['b', 'a', 'c'], dtype=object)
    print(f"\nString values: {str_values}")
    str_result = np.argsort(str_values)
    print(f"np.argsort on strings: {str_result}")
    print(f"np.argsort on strings shape: {str_result.shape}")
    print(f"np.argsort on strings ndim: {str_result.ndim}")

    # Test argsort on numbers for comparison
    num_values = np.array([2, 1, 3], dtype=object)
    print(f"\nNumber values: {num_values}")
    num_result = np.argsort(num_values)
    print(f"np.argsort on numbers: {num_result}")
    print(f"np.argsort on numbers shape: {num_result.shape}")
    print(f"np.argsort on numbers ndim: {num_result.ndim}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing _sort_mixed bug report")
    print("=" * 60)

    # Run the investigation first
    investigate_argsort_behavior()

    print("\n" + "=" * 60)
    print("Running test cases")
    print("=" * 60)

    results = []
    results.append(("Simple reproduction", test_simple_reproduction()))
    results.append(("Multiple tuples", test_multiple_tuples()))
    results.append(("Mixed with tuples", test_mixed_with_tuples()))
    results.append(("Without tuples", test_without_tuples()))
    results.append(("Empty tuple", test_empty_tuples()))
    results.append(("Multi-element tuples", test_multi_element_tuples()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    failed_count = sum(1 for _, passed in results if not passed)
    if failed_count > 0:
        print(f"\n{failed_count} test(s) failed!")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)