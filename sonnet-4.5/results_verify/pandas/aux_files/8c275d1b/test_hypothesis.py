from hypothesis import given, strategies as st, settings
from pandas.io.formats.format import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20, unique=True))
@settings(max_examples=1000)
def test_format_percentiles_unique_inputs_remain_unique(percentiles):
    """
    Property from docstring: "if any two elements of percentiles differ,
    they remain different after rounding"
    """
    formatted = format_percentiles(percentiles)
    assert len(formatted) == len(set(formatted)), \
        f"Unique inputs produced duplicate outputs: {percentiles} -> {formatted}"

if __name__ == "__main__":
    test_format_percentiles_unique_inputs_remain_unique()
    print("Test passed!")