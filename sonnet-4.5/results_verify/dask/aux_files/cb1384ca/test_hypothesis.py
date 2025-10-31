#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings
from dask.diagnostics.profile_visualize import unquote
import traceback

@given(
    expr=st.recursive(
        st.one_of(
            st.integers(),
            st.text(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        lambda children: st.tuples(
            st.sampled_from([tuple, list, set, dict]),
            st.lists(children, max_size=3)
        ),
        max_leaves=10
    )
)
@settings(max_examples=100)
def test_unquote_idempotence(expr):
    """Test that unquote is idempotent - applying it twice gives the same result as once."""
    try:
        once = unquote(expr)
        twice = unquote(once)
        assert once == twice, f"Failed idempotence: {expr} -> {once} -> {twice}"
    except IndexError as e:
        print(f"IndexError on input: {expr}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error on input: {expr}")
        print(f"Error: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Running Hypothesis property-based test...")
    try:
        test_unquote_idempotence()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()