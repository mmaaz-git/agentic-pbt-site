import pytest
from hypothesis import given, strategies as st, example
import scipy.constants as const
from scipy.constants import _codata


@given(st.sampled_from(list(const.physical_constants.keys())))
@example('muon Compton wavelength over 2 pi')  # Explicitly test the failing case from bug report
def test_find_searches_all_physical_constants(key):
    """All keys in physical_constants should be findable."""
    search_term = key.split()[0][:4]
    find_result = const.find(search_term)
    expected_matches = [k for k in const.physical_constants.keys()
                       if search_term.lower() in k.lower()]
    if key in expected_matches:
        assert key in find_result, f"Key '{key}' should be in find('{search_term}') results but was not found"


if __name__ == "__main__":
    test_find_searches_all_physical_constants()