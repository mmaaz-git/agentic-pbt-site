#!/usr/bin/env python3
"""Property-based tests using Hypothesis for format_percentiles"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, assume, settings
from pandas.io.formats.format import format_percentiles

print("Running property-based tests with Hypothesis...")
print("=" * 60)

# Test 1: Uniqueness property
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
@settings(max_examples=100)
def test_format_percentiles_uniqueness_property(percentiles):
    unique_percentiles = list(set(percentiles))
    assume(len(unique_percentiles) >= 2)

    try:
        result = format_percentiles(unique_percentiles)

        unique_results = set(result)
        assert len(unique_results) == len(unique_percentiles), \
            f"Uniqueness not preserved: {unique_percentiles} -> {result}"
        print(f"✓ Test passed for {unique_percentiles[:3]}..." if len(unique_percentiles) > 3 else f"✓ Test passed for {unique_percentiles}")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e} for input {unique_percentiles}")
        return False
    return True

# Test 2: Never rounds to 0% or 100% unless exactly 0 or 1
@given(st.lists(st.floats(min_value=1e-10, max_value=1.0 - 1e-10, allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
@settings(max_examples=100)
def test_format_percentiles_never_rounds_to_zero_or_hundred(percentiles):
    assume(all(0 < p < 1 for p in percentiles))

    try:
        result = format_percentiles(percentiles)

        for s in result:
            assert s != "0%", f"Rounded to 0% for input {percentiles}"
            assert s != "100%", f"Rounded to 100% for input {percentiles}"
        print(f"✓ Test passed for {percentiles[:3]}..." if len(percentiles) > 3 else f"✓ Test passed for {percentiles}")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e} for input {percentiles}")
        return False
    return True

print("\nTest 1: Uniqueness Property")
print("-" * 40)
try:
    test_format_percentiles_uniqueness_property()
    print("All tests passed for uniqueness property")
except Exception as e:
    print(f"Tests failed: {e}")

print("\n\nTest 2: Never Rounds to 0% or 100%")
print("-" * 40)
try:
    test_format_percentiles_never_rounds_to_zero_or_hundred()
    print("All tests passed for rounding property")
except Exception as e:
    print(f"Tests failed: {e}")

# Now let's test some specific edge cases mentioned in the bug report
print("\n\n" + "=" * 60)
print("Testing specific edge cases from bug report:")
print("=" * 60)

edge_cases = [
    [0.0, 5e-324],
    [0.0, 1.401298464324817e-45],
    [1e-10],
    [1e-15, 2e-15],
    [0.999999999, 1.0]
]

for case in edge_cases:
    print(f"\nInput: {case}")
    try:
        result = format_percentiles(case)
        print(f"Output: {result}")

        # Check uniqueness
        if len(set(case)) != len(set(result)):
            print(f"  ⚠️  Uniqueness violated!")

        # Check for improper 0% or 100%
        for i, (inp, out) in enumerate(zip(case, result)):
            if 0 < inp < 1 and (out == "0%" or out == "100%"):
                print(f"  ⚠️  Value {inp} incorrectly rounded to {out}")

        # Check for NaN
        if any("nan" in r.lower() for r in result):
            print(f"  ⚠️  NaN values in output!")

    except Exception as e:
        print(f"Error: {e}")