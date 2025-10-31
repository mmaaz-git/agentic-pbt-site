import numpy as np
from hypothesis import given, strategies as st, assume, settings
from pandas.io.formats.format import format_percentiles

@settings(max_examples=1000)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_format_percentiles_uniqueness_preservation(percentiles):
    unique_input = np.unique(percentiles)
    assume(len(unique_input) > 1)

    formatted = format_percentiles(percentiles)
    unique_input_formatted = format_percentiles(unique_input)
    unique_output = list(dict.fromkeys(unique_input_formatted))

    assert len(unique_output) == len(unique_input), \
        f"Unique percentiles should remain unique after formatting. " \
        f"Input had {len(unique_input)} unique values but output has {len(unique_output)}."

if __name__ == "__main__":
    test_format_percentiles_uniqueness_preservation()
    print("Test passed!")