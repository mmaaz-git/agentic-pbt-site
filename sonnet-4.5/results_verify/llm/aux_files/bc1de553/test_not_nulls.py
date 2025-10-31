#!/usr/bin/env python3
"""Test the not_nulls function bug report."""

import sys
import traceback

# First test the basic reproduction case
def test_basic_bug():
    """Test the basic reproduction case from the bug report."""
    print("Testing basic reproduction case...")

    # Import the function
    from llm.default_plugins.openai_models import not_nulls

    # Test case from bug report
    data = {'': None}
    try:
        result = not_nulls(data)
        print(f"ERROR: Function unexpectedly succeeded with result: {result}")
        return False
    except ValueError as e:
        print(f"SUCCESS: Got expected ValueError: {e}")
        return True
    except Exception as e:
        print(f"ERROR: Got unexpected exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_with_non_empty_dict():
    """Test with a non-empty dict with actual values."""
    print("\nTesting with non-empty dict...")

    from llm.default_plugins.openai_models import not_nulls

    data = {'key1': 'value1', 'key2': None, 'key3': 42}
    try:
        result = not_nulls(data)
        print(f"ERROR: Function unexpectedly succeeded with result: {result}")
        return False
    except ValueError as e:
        print(f"SUCCESS: Got expected ValueError: {e}")
        return True
    except Exception as e:
        print(f"ERROR: Got unexpected exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_with_hypothesis():
    """Test with the hypothesis test from the bug report."""
    print("\nTesting with Hypothesis...")

    try:
        from hypothesis import given, strategies as st
        from llm.default_plugins.openai_models import not_nulls

        @given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
        def test_not_nulls_removes_none_values(data):
            result = not_nulls(data)
            assert isinstance(result, dict)
            assert all(value is not None for value in result.values())

        # Run the hypothesis test
        test_not_nulls_removes_none_values()
        print("ERROR: Hypothesis test unexpectedly passed")
        return False
    except ValueError as e:
        print(f"SUCCESS: Hypothesis test failed with ValueError as expected: {e}")
        return True
    except AssertionError as e:
        print(f"ERROR: Hypothesis test failed with AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"ERROR: Hypothesis test failed with unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing not_nulls function bug report")
    print("=" * 60)

    all_passed = True

    # Run all tests
    all_passed = test_basic_bug() and all_passed
    all_passed = test_with_non_empty_dict() and all_passed
    all_passed = test_with_hypothesis() and all_passed

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests confirmed the bug exists!")
    else:
        print("Some tests did not behave as expected")
    print("=" * 60)