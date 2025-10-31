import numpy as np
from pandas.core import algorithms
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=50))
@example(['\x00'])  # The failing case mentioned in the bug report
def test_factorize_strings_round_trip(values):
    arr = np.array(values)
    codes, uniques = algorithms.factorize(arr)

    assert len(codes) == len(values)

    reconstructed = [uniques[code] if code >= 0 else None for code in codes]
    assert list(reconstructed) == list(values)

if __name__ == '__main__':
    test_factorize_strings_round_trip()