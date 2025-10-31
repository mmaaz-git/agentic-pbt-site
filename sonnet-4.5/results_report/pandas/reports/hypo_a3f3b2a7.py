from hypothesis import given, strategies as st
from pandas.io.formats.format import _trim_zeros_complex


@given(st.lists(st.tuples(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)
), min_size=1, max_size=10))
def test_trim_zeros_complex_preserves_parentheses(float_pairs):
    values = [complex(r, i) for r, i in float_pairs]
    str_complexes = [str(v) for v in values]
    trimmed = _trim_zeros_complex(str_complexes)

    for original, result in zip(str_complexes, trimmed):
        if original.endswith(')'):
            assert result.endswith(')'), f"Lost closing parenthesis: {original} -> {result}"


if __name__ == "__main__":
    # Run the test
    test_trim_zeros_complex_preserves_parentheses()