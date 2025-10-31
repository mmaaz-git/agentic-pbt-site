#!/usr/bin/env python3
"""Test the unquote bug reported for dask.diagnostics.profile_visualize.unquote"""

# First, verify the hypothesis test case
from hypothesis import given, strategies as st
from dask.core import istask
from dask.diagnostics.profile_visualize import unquote

# Test the hypothesis property test
@given(st.just((dict, [])))
def test_unquote_handles_empty_dict_task(expr):
    """Test from the bug report - should pass if bug is fixed, fail if not"""
    assert istask(expr)
    result = unquote(expr)
    assert result == {}

# Manual reproduction of the bug
def test_manual_reproduction():
    """Direct reproduction of the bug"""
    from dask.diagnostics.profile_visualize import unquote

    expr = (dict, [])

    # First verify it's a valid task
    from dask.core import istask
    print(f"Is (dict, []) a valid dask task? {istask(expr)}")

    # Now try to unquote it - this should crash with IndexError
    try:
        result = unquote(expr)
        print(f"Result: {result}")
        return True
    except IndexError as e:
        print(f"IndexError occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing manual reproduction of the bug...")
    print("=" * 60)
    success = test_manual_reproduction()

    print("\n" + "=" * 60)
    print("Testing hypothesis test case...")
    print("=" * 60)
    try:
        test_unquote_handles_empty_dict_task()
        print("Hypothesis test passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")
        import traceback
        traceback.print_exc()