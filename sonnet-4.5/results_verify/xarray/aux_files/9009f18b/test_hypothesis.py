#!/usr/bin/env python3
"""Run the hypothesis tests from the bug report"""

from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(st.builds(AlwaysGreaterThan))
def test_always_greater_than_irreflexivity(agt):
    """Test that AlwaysGreaterThan satisfies irreflexivity: agt > agt should be False"""
    result = agt > agt
    print(f"Testing: agt > agt = {result}")
    assert not (agt > agt), "AlwaysGreaterThan violates irreflexivity: agt > agt is True"


@settings(max_examples=100)
@given(st.builds(AlwaysGreaterThan), st.builds(AlwaysGreaterThan))
def test_trichotomy_law(agt1, agt2):
    """Test trichotomy: exactly one of a < b, a == b, a > b should be true"""
    less = agt1 < agt2
    equal = agt1 == agt2
    greater = agt1 > agt2

    true_count = sum([less, equal, greater])
    print(f"Testing trichotomy: less={less}, equal={equal}, greater={greater}, count={true_count}")
    assert true_count == 1, f"Trichotomy violated: {true_count} conditions are true"


@settings(max_examples=100)
@given(st.builds(AlwaysLessThan))
def test_always_less_than_irreflexivity(alt):
    """Test that AlwaysLessThan satisfies irreflexivity: alt < alt should be False"""
    result = alt < alt
    print(f"Testing: alt < alt = {result}")
    assert not (alt < alt), "AlwaysLessThan violates irreflexivity: alt < alt is True"


if __name__ == "__main__":
    print("Testing AlwaysGreaterThan irreflexivity...")
    try:
        test_always_greater_than_irreflexivity()
        print("✓ AlwaysGreaterThan irreflexivity test passed")
    except AssertionError as e:
        print(f"✗ AlwaysGreaterThan irreflexivity test failed: {e}")

    print("\nTesting AlwaysGreaterThan trichotomy...")
    try:
        test_trichotomy_law()
        print("✓ AlwaysGreaterThan trichotomy test passed")
    except AssertionError as e:
        print(f"✗ AlwaysGreaterThan trichotomy test failed: {e}")

    print("\nTesting AlwaysLessThan irreflexivity...")
    try:
        test_always_less_than_irreflexivity()
        print("✓ AlwaysLessThan irreflexivity test passed")
    except AssertionError as e:
        print(f"✗ AlwaysLessThan irreflexivity test failed: {e}")