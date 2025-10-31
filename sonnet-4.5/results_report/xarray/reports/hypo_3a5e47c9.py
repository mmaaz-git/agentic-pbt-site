from hypothesis import given, strategies as st
from xarray.core.utils import OrderedSet

@given(st.lists(st.integers()), st.integers())
def test_orderedset_discard_never_raises(initial_values, value_to_discard):
    """
    Property: discard() should never raise an error, whether the element
    exists or not. This is the core contract of MutableSet.discard().
    """
    os = OrderedSet(initial_values)
    os.discard(value_to_discard)
    # If we reach here without raising, the test passes

# Run the test
if __name__ == "__main__":
    test_orderedset_discard_never_raises()