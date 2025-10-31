#!/usr/bin/env python3
"""Test the reported bug about InfinityType violating trichotomy law"""

import pandas.util.version as pv

# Test from the bug report
def test_infinity_equals_itself():
    inf = pv.Infinity
    print("Testing Infinity comparison with itself:")
    print(f"inf == inf: {inf == inf}")
    print(f"inf != inf: {inf != inf}")
    print(f"inf < inf: {inf < inf}")
    print(f"inf > inf: {inf > inf}")
    print(f"inf <= inf: {inf <= inf}")
    print(f"inf >= inf: {inf >= inf}")

    # The assertions from the bug report
    try:
        assert inf == inf
        print("✓ inf == inf assertion passed")
    except AssertionError:
        print("✗ inf == inf assertion FAILED")

    try:
        assert not (inf != inf)
        print("✓ not (inf != inf) assertion passed")
    except AssertionError:
        print("✗ not (inf != inf) assertion FAILED")

    try:
        assert not (inf < inf)
        print("✓ not (inf < inf) assertion passed")
    except AssertionError:
        print("✗ not (inf < inf) assertion FAILED")

    try:
        assert not (inf > inf)
        print("✓ not (inf > inf) assertion passed")
    except AssertionError:
        print("✗ not (inf > inf) assertion FAILED - THIS IS THE BUG!")

    try:
        assert inf <= inf
        print("✓ inf <= inf assertion passed")
    except AssertionError:
        print("✗ inf <= inf assertion FAILED")

    try:
        assert inf >= inf
        print("✓ inf >= inf assertion passed")
    except AssertionError:
        print("✗ inf >= inf assertion FAILED")

# Test trichotomy law
def test_trichotomy():
    inf = pv.Infinity
    print("\nTesting trichotomy law (exactly one of <, ==, > should be True):")

    is_equal = inf == inf
    is_less = inf < inf
    is_greater = inf > inf

    print(f"inf == inf: {is_equal}")
    print(f"inf < inf: {is_less}")
    print(f"inf > inf: {is_greater}")

    count_true = sum([is_equal, is_less, is_greater])
    print(f"\nNumber of True comparisons: {count_true}")
    print(f"Expected: 1 (exactly one should be True)")
    print(f"Result: {'PASS' if count_true == 1 else 'FAIL - TRICHOTOMY VIOLATED'}")

# Test with other values
def test_with_other_values():
    inf = pv.Infinity
    print("\nTesting Infinity with other values:")
    print(f"inf > 999999: {inf > 999999}")
    print(f"inf > 'string': {inf > 'string'}")
    print(f"inf > None: {inf > None}")
    print(f"inf == 999999: {inf == 999999}")

    # Test NegativeInfinity too
    neg_inf = pv.NegativeInfinity
    print(f"\nneg_inf < inf: {neg_inf < inf}")
    print(f"inf > neg_inf: {inf > neg_inf}")

if __name__ == "__main__":
    test_infinity_equals_itself()
    test_trichotomy()
    test_with_other_values()