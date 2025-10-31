#!/usr/bin/env python3
"""Test file to reproduce the check_array_indexer bug."""

import numpy as np
import pandas as pd
from pandas.api.indexers import check_array_indexer

def test_property_based():
    """Run the property-based test from the bug report."""
    from hypothesis import assume, given, settings, strategies as st

    @settings(max_examples=10)
    @given(
        array_len=st.integers(min_value=0, max_value=100),
        indexer_data=st.lists(st.booleans(), max_size=100)
    )
    def test_check_array_indexer_idempotence_boolean(array_len, indexer_data):
        array = pd.array(range(array_len))
        indexer_len = len(indexer_data)
        assume(indexer_len == array_len)

        indexer = pd.array(indexer_data)

        result1 = check_array_indexer(array, indexer)
        result2 = check_array_indexer(array, result1)

        np.testing.assert_array_equal(result1, result2)

    # Run the test
    try:
        test_check_array_indexer_idempotence_boolean()
        print("Property-based test passed")
    except Exception as e:
        print(f"Property-based test failed: {e}")

def test_specific_case():
    """Test the specific failing case mentioned in the bug report."""
    print("\n=== Testing specific case: array_len=0, indexer_data=[] ===")

    # Create empty array and indexer
    array = pd.array([], dtype='int64')
    indexer = pd.array([], dtype='bool')

    print(f"Array: {array} (dtype: {array.dtype})")
    print(f"Indexer: {indexer} (dtype: {indexer.dtype})")

    try:
        result1 = check_array_indexer(array, indexer)
        print(f"First call succeeded: {result1} (dtype: {result1.dtype})")

        result2 = check_array_indexer(array, result1)
        print(f"Second call succeeded: {result2} (dtype: {result2.dtype})")

        np.testing.assert_array_equal(result1, result2)
        print("Idempotence check passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_empty_comparisons():
    """Test the main comparison from the bug report."""
    print("\n=== Testing empty list vs empty pandas array ===")

    array = pd.array([1, 2, 3])

    # Test with empty list
    print("\n1. Testing with empty Python list:")
    empty_list = []
    try:
        result1 = check_array_indexer(array, empty_list)
        print(f"   Empty list works: {result1} (dtype: {result1.dtype})")
    except Exception as e:
        print(f"   Empty list fails: {e}")

    # Test with empty pandas array
    print("\n2. Testing with empty pandas array:")
    empty_pandas_array = pd.array([])
    print(f"   Empty pandas array dtype: {empty_pandas_array.dtype}")
    try:
        result2 = check_array_indexer(array, empty_pandas_array)
        print(f"   Empty pandas array works: {result2} (dtype: {result2.dtype})")
    except Exception as e:
        print(f"   Empty pandas array fails: {e}")

    # Test with empty numpy array
    print("\n3. Testing with empty numpy array (int):")
    empty_numpy_int = np.array([], dtype=np.int64)
    try:
        result3 = check_array_indexer(array, empty_numpy_int)
        print(f"   Empty numpy int array works: {result3} (dtype: {result3.dtype})")
    except Exception as e:
        print(f"   Empty numpy int array fails: {e}")

    # Test with empty numpy array (bool)
    print("\n4. Testing with empty numpy array (bool):")
    empty_numpy_bool = np.array([], dtype=bool)
    try:
        result4 = check_array_indexer(array, empty_numpy_bool)
        print(f"   Empty numpy bool array works: {result4} (dtype: {result4.dtype})")
    except Exception as e:
        print(f"   Empty numpy bool array fails: {e}")

    # Test with empty numpy array (float)
    print("\n5. Testing with empty numpy array (float):")
    empty_numpy_float = np.array([], dtype=np.float64)
    try:
        result5 = check_array_indexer(array, empty_numpy_float)
        print(f"   Empty numpy float array works: {result5} (dtype: {result5.dtype})")
    except Exception as e:
        print(f"   Empty numpy float array fails: {e}")

def investigate_pd_array_empty():
    """Investigate what pd.array([]) returns."""
    print("\n=== Investigating pd.array([]) behavior ===")

    empty_pd = pd.array([])
    print(f"pd.array([]) creates: {empty_pd}")
    print(f"  dtype: {empty_pd.dtype}")
    print(f"  type: {type(empty_pd)}")
    print(f"  length: {len(empty_pd)}")

    # Check if it's array-like
    from pandas.core.dtypes.common import is_array_like
    print(f"  is_array_like: {is_array_like(empty_pd)}")

    # Try different empty creations
    print("\nOther empty array creations:")
    empty_bool = pd.array([], dtype='bool')
    print(f"pd.array([], dtype='bool'): {empty_bool} (dtype: {empty_bool.dtype})")

    empty_int = pd.array([], dtype='int64')
    print(f"pd.array([], dtype='int64'): {empty_int} (dtype: {empty_int.dtype})")

if __name__ == "__main__":
    # Run all tests
    test_property_based()
    test_specific_case()
    test_empty_comparisons()
    investigate_pd_array_empty()