from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.just(None))
def test_argsort_defaults_duplicate_assignment(expected):
    actual = ARGSORT_DEFAULTS['kind']
    assert actual == expected

if __name__ == "__main__":
    test_argsort_defaults_duplicate_assignment()
    print("Test passed: ARGSORT_DEFAULTS['kind'] is None")