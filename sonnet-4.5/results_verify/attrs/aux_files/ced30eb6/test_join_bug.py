from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db


@given(st.integers())
def test_join_error_message_with_invalid_type(invalid_input):
    """
    Property: Error messages should be displayable without crashing.

    This test verifies that when join() is called with an invalid type,
    it should raise a TypeError with a proper error message, not crash
    with an AttributeError.
    """
    bag = db.from_sequence([1, 2, 3])

    try:
        bag.join(invalid_input, lambda x: x)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Joined argument must be" in str(e)
    except AttributeError as e:
        if "'type' object has no attribute '__name'" in str(e):
            raise AssertionError(
                f"Bug found! Error message construction failed with AttributeError: {e}"
            )

if __name__ == "__main__":
    # Run a simple test case without the decorator
    bag = db.from_sequence([1, 2, 3])

    try:
        bag.join(42, lambda x: x)
        print("ERROR: Should have raised an exception")
    except TypeError as e:
        print(f"TypeError (expected): {e}")
    except AttributeError as e:
        if "'type' object has no attribute '__name'" in str(e):
            print(f"Bug found! Error message construction failed with AttributeError: {e}")
        else:
            print(f"Unexpected AttributeError: {e}")