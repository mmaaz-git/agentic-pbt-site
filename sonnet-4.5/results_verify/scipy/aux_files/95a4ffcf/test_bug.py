#!/usr/bin/env python3
"""Test the reported bug with scipy.constants.find()"""

import scipy.constants as sc

# First, let's run the reproduction code
print("=== Reproduction Code ===")
all_keys = set(sc.physical_constants.keys())
find_none_keys = set(sc.find(None))
find_empty_keys = set(sc.find(''))

print(f"Total keys in physical_constants: {len(all_keys)}")
print(f"Keys returned by find(None): {len(find_none_keys)}")
print(f"Keys returned by find(''): {len(find_empty_keys)}")
print(f"Missing from find(None): {len(all_keys - find_none_keys)}")

missing = all_keys - find_none_keys
print("\nSample missing keys:")
for key in list(missing)[:10]:
    print(f"  '{key}'")

# Now let's run the property-based tests
print("\n=== Property-Based Tests ===")

def test_find_none_returns_all_keys():
    results = sc.find(None)
    expected = set(sc.physical_constants.keys())
    actual = set(results)
    try:
        assert actual == expected
        print("test_find_none_returns_all_keys: PASSED")
        return True
    except AssertionError:
        print(f"test_find_none_returns_all_keys: FAILED")
        print(f"  Expected {len(expected)} keys, got {len(actual)} keys")
        print(f"  Missing {len(expected - actual)} keys")
        return False

def test_find_empty_string_returns_all_keys():
    results = sc.find('')
    expected = set(sc.physical_constants.keys())
    actual = set(results)
    try:
        assert actual == expected
        print("test_find_empty_string_returns_all_keys: PASSED")
        return True
    except AssertionError:
        print(f"test_find_empty_string_returns_all_keys: FAILED")
        print(f"  Expected {len(expected)} keys, got {len(actual)} keys")
        print(f"  Missing {len(expected - actual)} keys")
        return False

test_find_none_returns_all_keys()
test_find_empty_string_returns_all_keys()

# Let's also check what _current_constants contains
print("\n=== Additional Investigation ===")
try:
    from scipy.constants._codata import _current_constants
    print(f"Keys in _current_constants: {len(_current_constants.keys())}")
    print(f"Keys in physical_constants: {len(sc.physical_constants.keys())}")

    # Check if the missing keys are the difference
    current_keys = set(_current_constants.keys())
    all_keys = set(sc.physical_constants.keys())
    diff = all_keys - current_keys
    print(f"Difference between physical_constants and _current_constants: {len(diff)} keys")

    # Verify these are the same missing keys
    if diff == missing:
        print("CONFIRMED: The missing keys are exactly those not in _current_constants")
    else:
        print("WARNING: The missing keys don't match _current_constants difference")
except ImportError:
    print("Could not import _current_constants for investigation")