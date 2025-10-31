#!/usr/bin/env python3
"""Test the reported bug in pandas.core.util.hashing.hash_array"""

import numpy as np
from pandas.core.util.hashing import hash_array
from hypothesis import given, strategies as st, settings

# First, reproduce the exact bug report
def test_exact_reproduction():
    """Reproduce the exact case reported in the bug"""
    values = ['', '\x00']
    arr = np.array(values, dtype=object)

    hash_with_categorize = hash_array(arr, categorize=True)
    hash_without_categorize = hash_array(arr, categorize=False)

    print("Test 1: Exact reproduction of bug report")
    print("Input values:", repr(values))
    print("Hash with categorize=True: ", hash_with_categorize)
    print("Hash with categorize=False:", hash_without_categorize)
    print("Equal?", np.array_equal(hash_with_categorize, hash_without_categorize))
    print()

    return np.array_equal(hash_with_categorize, hash_without_categorize)

# Run the hypothesis test
@given(st.lists(st.text(min_size=0, max_size=100), min_size=1))
@settings(max_examples=500)
def test_hash_array_categorize_consistency(values):
    """Property-based test from the bug report"""
    arr = np.array(values, dtype=object)
    hash_with_categorize = hash_array(arr, categorize=True)
    hash_without_categorize = hash_array(arr, categorize=False)
    assert np.array_equal(hash_with_categorize, hash_without_categorize), \
        f"categorize parameter should be optimization only, not change result. Failed on: {repr(values)}"

# Additional test cases to understand the behavior
def test_additional_cases():
    """Test some additional edge cases"""
    test_cases = [
        ['', ''],  # Two empty strings
        ['\x00', '\x00'],  # Two null bytes
        ['a', 'b'],  # Different regular strings
        ['a', 'a'],  # Same regular strings
        ['', ' '],  # Empty string vs space
        ['', 'a'],  # Empty string vs non-empty
        ['\x00', 'a'],  # Null byte vs regular char
        ['', '\x00', 'a'],  # Mix of all three
    ]

    print("Test 2: Additional test cases")
    for values in test_cases:
        arr = np.array(values, dtype=object)
        hash_with = hash_array(arr, categorize=True)
        hash_without = hash_array(arr, categorize=False)
        equal = np.array_equal(hash_with, hash_without)
        if not equal:
            print(f"  Mismatch for {repr(values)}:")
            print(f"    With categorize:    {hash_with}")
            print(f"    Without categorize: {hash_without}")
    print()

# Test to understand what factorize does
def test_factorize_behavior():
    """Check how pandas.factorize handles empty string and null byte"""
    from pandas import factorize

    print("Test 3: Understanding pandas.factorize behavior")
    values = ['', '\x00']
    arr = np.array(values, dtype=object)

    codes, categories = factorize(arr, sort=False)
    print(f"Input: {repr(values)}")
    print(f"Codes from factorize: {codes}")
    print(f"Categories from factorize: {repr(list(categories))}")
    print(f"Are '' and '\\x00' treated as same category? {codes[0] == codes[1]}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing pandas.core.util.hashing.hash_array categorize bug")
    print("=" * 60)
    print()

    # Run the exact reproduction
    result1 = test_exact_reproduction()

    # Run additional tests
    test_additional_cases()

    # Understand factorize
    test_factorize_behavior()

    # Run hypothesis test
    print("Test 4: Running Hypothesis property-based test")
    print("(This may find additional failures)")
    try:
        test_hash_array_categorize_consistency.hypothesis.inner_test()
        print("Hypothesis test passed!")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")