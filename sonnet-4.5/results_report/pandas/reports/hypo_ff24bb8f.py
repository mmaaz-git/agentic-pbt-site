from hypothesis import given, strategies as st

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.just(None))
def test_argsort_defaults_kind_value(dummy):
    """Test that ARGSORT_DEFAULTS['kind'] is None.

    This test passes, but reveals dead code: line 138 sets
    ARGSORT_DEFAULTS['kind'] = 'quicksort' which is immediately
    overwritten by line 140 setting it to None.
    """
    assert ARGSORT_DEFAULTS["kind"] is None

# Run the test
test_argsort_defaults_kind_value()