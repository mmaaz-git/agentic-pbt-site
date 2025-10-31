import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.algorithms import factorize

@given(st.lists(st.text(min_size=0, max_size=10)))
@settings(max_examples=1000)
def test_factorize_roundtrip_strings(values):
    values_array = np.array(values, dtype=object)
    codes, uniques = factorize(values_array)

    reconstructed = uniques.take(codes[codes >= 0])
    original_non_nan = values_array[codes >= 0]

    assert len(reconstructed) == len(original_non_nan)
    assert np.array_equal(reconstructed, original_non_nan)

if __name__ == "__main__":
    test_factorize_roundtrip_strings()