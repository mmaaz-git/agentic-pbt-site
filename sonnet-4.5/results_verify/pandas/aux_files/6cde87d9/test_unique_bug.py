#!/usr/bin/env python3
"""Test script to reproduce the unique() bug with null characters"""

import numpy as np
from pandas.core.algorithms import unique, factorize

print("=" * 60)
print("Testing pandas.core.algorithms.unique with null characters")
print("=" * 60)

# Test case 1: Empty string vs null character
print("\n1. Testing empty string '' vs null character '\\x00':")
values = np.array(['', '\x00'], dtype=object)
uniques = unique(values)

print(f"Input values: {[repr(v) for v in values]}")
print(f"Python set: {set(values)}")
print(f"Set length: {len(set(values))}")
print(f"unique() result: {[repr(v) for v in uniques]}")
print(f"unique() length: {len(uniques)}")
print(f"Expected length: 2, Got: {len(uniques)}")

# Verify they are actually different in Python
print(f"\nAre '' and '\\x00' equal in Python? {'' == '\x00'}")
print(f"Hash of '': {hash('')}")
print(f"Hash of '\\x00': {hash('\x00')}")

# Test case 2: Different null character strings
print("\n2. Testing '\\x00' vs '\\x00\\x00':")
values2 = np.array(['\x00', '\x00\x00'], dtype=object)
uniques2 = unique(values2)

print(f"Input values: {[repr(v) for v in values2]}")
print(f"Python set: {set(values2)}")
print(f"Set length: {len(set(values2))}")
print(f"unique() result: {[repr(v) for v in uniques2]}")
print(f"unique() length: {len(uniques2)}")
print(f"Expected length: 2, Got: {len(uniques2)}")

# Test case 3: Mixed with regular string
print("\n3. Testing '', '\\x00', 'a':")
values3 = np.array(['', '\x00', 'a'], dtype=object)
uniques3 = unique(values3)

print(f"Input values: {[repr(v) for v in values3]}")
print(f"Python set: {set(values3)}")
print(f"Set length: {len(set(values3))}")
print(f"unique() result: {[repr(v) for v in uniques3]}")
print(f"unique() length: {len(uniques3)}")
print(f"Expected length: 3, Got: {len(uniques3)}")

# Test case 4: Regular strings work fine
print("\n4. Testing regular strings 'a', 'b', 'c':")
values4 = np.array(['a', 'b', 'c', 'a'], dtype=object)
uniques4 = unique(values4)

print(f"Input values: {[repr(v) for v in values4]}")
print(f"Python set: {set(values4)}")
print(f"Set length: {len(set(values4))}")
print(f"unique() result: {[repr(v) for v in uniques4]}")
print(f"unique() length: {len(uniques4)}")

# Test relationship with factorize
print("\n5. Comparing with factorize() function:")
values5 = np.array(['', '\x00'], dtype=object)
codes, uniques_from_factorize = factorize(values5)

print(f"unique() result: {[repr(v) for v in unique(values5)]}")
print(f"factorize() uniques: {[repr(v) for v in uniques_from_factorize]}")
print(f"factorize() codes: {codes}")

# Run the hypothesis test case
print("\n" + "=" * 60)
print("Running Hypothesis test")
print("=" * 60)

from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=0, max_size=10)))
@settings(max_examples=1000)
def test_unique_returns_all_distinct_values(values):
    values_array = np.array(values, dtype=object)
    uniques = unique(values_array)

    assert len(uniques) == len(set(values)), \
        f"Failed on {values}: unique returned {len(uniques)} values but set has {len(set(values))}"
    assert set(uniques) == set(values), \
        f"Failed on {values}: unique returned {set(uniques)} but expected {set(values)}"

try:
    test_unique_returns_all_distinct_values()
    print("Hypothesis test PASSED")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Hypothesis test ERROR: {e}")

# Explicitly test the failing case from the bug report
print("\nExplicitly testing the reported failing case ['', '\\x00']:")
try:
    test_values = ['', '\x00']
    values_array = np.array(test_values, dtype=object)
    uniques = unique(values_array)

    assert len(uniques) == len(set(test_values)), \
        f"unique() returned {len(uniques)} values but set has {len(set(test_values))}"
    assert set(uniques) == set(test_values), \
        f"unique() returned {set(uniques)} but expected {set(test_values)}"
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")