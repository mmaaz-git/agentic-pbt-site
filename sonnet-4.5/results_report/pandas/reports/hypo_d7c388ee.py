import pytest
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, SORT_DEFAULTS


def test_argsort_sort_defaults_consistency():
    """
    Property: ARGSORT_DEFAULTS and SORT_DEFAULTS should have consistent 'kind' values
    since both numpy.sort and numpy.argsort have the same default sorting algorithm.
    """
    assert ARGSORT_DEFAULTS['kind'] == SORT_DEFAULTS['kind'], \
        f"ARGSORT_DEFAULTS and SORT_DEFAULTS should have the same 'kind' default, but got {ARGSORT_DEFAULTS['kind']!r} != {SORT_DEFAULTS['kind']!r}"

if __name__ == "__main__":
    test_argsort_sort_defaults_consistency()