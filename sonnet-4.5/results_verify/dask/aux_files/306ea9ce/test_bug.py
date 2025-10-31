#!/usr/bin/env python3
"""Test the reported bug in dask.diagnostics.profile_visualize.unquote"""

# First, let's test if dask is available
try:
    from dask.diagnostics.profile_visualize import unquote
    print("Successfully imported unquote function from dask")
except ImportError as e:
    print(f"Failed to import: {e}")
    import sys
    sys.exit(1)

# Test 1: Reproduce the exact bug case
print("\n=== Test 1: Reproduce the reported bug ===")
try:
    task = (dict, [])
    result = unquote(task)
    print(f"Result for (dict, []): {result}")
    print("No error occurred - bug might be fixed or report is incorrect")
except IndexError as e:
    print(f"IndexError occurred as reported: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Different error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Test with non-empty list
print("\n=== Test 2: Test with non-empty dict task ===")
try:
    task = (dict, [['key1', 'value1'], ['key2', 'value2']])
    result = unquote(task)
    print(f"Result for non-empty dict task: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Test 3: Run the Hypothesis test
print("\n=== Test 3: Run the Hypothesis property test ===")
try:
    from hypothesis import given, strategies as st

    @given(st.lists(st.tuples(st.text(), st.integers())))
    def test_unquote_dict_with_tuples(pairs):
        task = (dict, pairs)
        result = unquote(task)
        expected = dict(pairs)
        assert result == expected, f"Expected {expected}, got {result}"

    # Run the test with the specific failing case
    print("Testing with empty list (pairs=[])")
    test_unquote_dict_with_tuples([])
    print("Empty list test passed")

    # Run a few more examples
    print("Testing with some example pairs")
    test_unquote_dict_with_tuples([("a", 1), ("b", 2)])
    print("Non-empty pairs test passed")

except ImportError:
    print("Hypothesis not available, skipping property test")
except Exception as e:
    print(f"Property test failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing complete ===")