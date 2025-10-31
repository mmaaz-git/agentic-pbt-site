from hypothesis import given, strategies as st, assume
from pandas.io.formats.format import _trim_zeros_float


@given(
    num_strs=st.lists(
        st.from_regex(r'^\s*[\+-]?[0-9]+\.[0-9]+0$', fullmatch=True),
        min_size=2,
        max_size=10
    )
)
def test_trim_zeros_float_trims_uniformly(num_strs):
    assume(all('.' in s for s in num_strs))

    result = _trim_zeros_float(num_strs)

    decimal_lengths = []
    for r in result:
        if '.' in r:
            decimal_part = r.split('.')[-1].strip()
            decimal_lengths.append(len(decimal_part))

    if len(decimal_lengths) > 1:
        assert len(set(decimal_lengths)) == 1, f"Unequal decimal lengths: {decimal_lengths}"

# Run the test
if __name__ == "__main__":
    test_trim_zeros_float_trims_uniformly()