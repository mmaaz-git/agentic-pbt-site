#!/usr/bin/env python3
"""Test the reported bug in pandas.io.common.dedup_names"""

import sys
import traceback
from pandas.io.common import dedup_names

def test_case_1():
    """Test with empty string names"""
    try:
        result = dedup_names(['', ''], is_potential_multiindex=True)
        print(f"Test 1 passed: {result}")
        return True
    except AssertionError as e:
        print(f"Test 1 failed with AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Test 1 failed with unexpected error: {e}")
        traceback.print_exc()
        return False

def test_case_2():
    """Test with duplicate string names"""
    try:
        result = dedup_names(['x', 'x'], is_potential_multiindex=True)
        print(f"Test 2 passed: {result}")
        return True
    except AssertionError as e:
        print(f"Test 2 failed with AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Test 2 failed with unexpected error: {e}")
        traceback.print_exc()
        return False

def test_case_3():
    """Test with unique string names"""
    try:
        result = dedup_names(['a', 'b', 'c'], is_potential_multiindex=True)
        print(f"Test 3 passed: {result}")
        return True
    except AssertionError as e:
        print(f"Test 3 failed with AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Test 3 failed with unexpected error: {e}")
        traceback.print_exc()
        return False

def test_case_4():
    """Test with tuples when is_potential_multiindex=True"""
    try:
        result = dedup_names([('a',), ('a',)], is_potential_multiindex=True)
        print(f"Test 4 passed: {result}")
        return True
    except Exception as e:
        print(f"Test 4 failed: {e}")
        traceback.print_exc()
        return False

def test_case_5():
    """Test with strings when is_potential_multiindex=False"""
    try:
        result = dedup_names(['x', 'x'], is_potential_multiindex=False)
        print(f"Test 5 passed: {result}")
        return True
    except Exception as e:
        print(f"Test 5 failed: {e}")
        traceback.print_exc()
        return False

def test_hypothesis():
    """Test the property-based test from the bug report"""
    from hypothesis import given, strategies as st
    import pandas.io.common as common

    @given(st.lists(st.text()), st.booleans())
    def test_dedup_names_preserves_length(names, is_potential_multiindex):
        result = common.dedup_names(names, is_potential_multiindex)
        assert len(result) == len(names)

    try:
        # Run the property test
        test_dedup_names_preserves_length()
        print("Hypothesis test passed all cases")
        return True
    except Exception as e:
        print(f"Hypothesis test failed: {e}")
        traceback.print_exc()
        return False

def test_specific_hypothesis_case():
    """Test the specific failing case mentioned in the bug report"""
    try:
        result = dedup_names(['', ''], is_potential_multiindex=True)
        print(f"Specific hypothesis case passed: {result}")
        return True
    except AssertionError as e:
        print(f"Specific hypothesis case failed with AssertionError")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Specific hypothesis case failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing pandas.io.common.dedup_names bug report")
    print("=" * 60)

    print("\n1. Testing with empty string names and is_potential_multiindex=True:")
    test_case_1()

    print("\n2. Testing with duplicate string names and is_potential_multiindex=True:")
    test_case_2()

    print("\n3. Testing with unique string names and is_potential_multiindex=True:")
    test_case_3()

    print("\n4. Testing with tuples when is_potential_multiindex=True:")
    test_case_4()

    print("\n5. Testing with strings when is_potential_multiindex=False:")
    test_case_5()

    print("\n6. Testing specific hypothesis failing case:")
    test_specific_hypothesis_case()

    print("\n7. Running hypothesis property test:")
    test_hypothesis()