#!/usr/bin/env python3
"""Reproduce the reported bug in dask.utils.ndeepmap"""

from hypothesis import given, strategies as st, settings
from dask.utils import ndeepmap


def identity(x):
    return x


def inc(x):
    return x + 1


# Test the property-based test from the bug report
@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_ndeepmap_n0_loses_data(lst):
    """
    Property: ndeepmap should not silently discard data.

    When n=0, the current implementation calls func(seq[0]),
    which loses all elements after the first.
    """
    result = ndeepmap(0, identity, lst)

    # The bug: result only depends on first element
    assert result == lst[0]

    # Demonstrate data loss: changing other elements doesn't affect result
    modified_lst = [lst[0]] + [999] * (len(lst) - 1)
    modified_result = ndeepmap(0, identity, modified_lst)

    # This assertion passes, proving elements lst[1:] are ignored
    assert result == modified_result


# Manual reproduction examples
def test_manual_reproduction():
    print("=== Manual Bug Reproduction ===\n")

    # Example 1: Simple case
    input_list = [10, 20, 30]
    result = ndeepmap(0, identity, input_list)
    print(f"Input:  {input_list}")
    print(f"Result: {result}")
    print(f"Expected (if no data loss): {input_list} or func(input_list)")
    print(f"Actual: Only first element {input_list[0]} is returned")
    assert result == 10  # Only the first element
    print("✓ Bug confirmed: Only first element is processed\n")

    # Example 2: Verify other elements are ignored
    input_modified = [10, 999, 999]
    result_modified = ndeepmap(0, identity, input_modified)
    assert result == result_modified
    print(f"Modified input: {input_modified}")
    print(f"Modified result: {result_modified}")
    print("✓ Bug confirmed: Changing elements after first has no effect\n")

    # Example 3: With a transformation function
    input_list2 = [1, 2, 3]
    result2 = ndeepmap(0, inc, input_list2)
    print(f"Input with inc function: {input_list2}")
    print(f"Result: {result2}")
    print(f"Expected (if applied to all): [2, 3, 4] or inc([1,2,3])")
    print(f"Actual: inc(1) = 2")
    assert result2 == 2
    print("✓ Bug confirmed: Function only applied to first element\n")

    # Example 4: Check what happens with empty list
    try:
        empty_result = ndeepmap(0, identity, [])
        print(f"Empty list result: {empty_result}")
    except Exception as e:
        print(f"Empty list raises: {type(e).__name__}: {e}")

    # Example 5: Check negative n
    neg_input = [1, 2, 3]
    neg_result = ndeepmap(-1, identity, neg_input)
    print(f"\nNegative n=-1 with {neg_input}: {neg_result}")
    assert neg_result == 1  # Still only first element
    print("✓ Negative n also loses data")


def test_existing_behavior():
    """Test the existing test case to understand intended behavior"""
    print("\n=== Existing Test Cases ===\n")

    # From test_utils.py
    L = 1
    result = ndeepmap(0, inc, L)
    print(f"ndeepmap(0, inc, 1) = {result}")
    assert result == 2

    L = [1]
    result = ndeepmap(0, inc, L)
    print(f"ndeepmap(0, inc, [1]) = {result}")
    assert result == 2
    print("✓ Existing tests only use single elements")


if __name__ == "__main__":
    # Run manual tests
    test_manual_reproduction()
    test_existing_behavior()

    # Run property-based test
    print("\n=== Running Property-Based Test ===")
    try:
        test_ndeepmap_n0_loses_data()
        print("✗ Property test should have found counterexamples")
    except AssertionError:
        print("✗ Property test failed (as expected - bug exists)")

    # Run with a specific example
    print("\nTesting specific example [1, 2, 3]:")
    test_ndeepmap_n0_loses_data([1, 2, 3])
    print("✓ Bug reproduced with [1, 2, 3]")