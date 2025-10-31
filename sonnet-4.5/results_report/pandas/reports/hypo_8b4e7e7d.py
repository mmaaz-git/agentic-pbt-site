from hypothesis import given, strategies as st, settings
from pandas.io.formats.css import CSSResolver
import re


@given(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_non_integer_pt_values_formatting(pt_val):
    resolver = CSSResolver()
    result = resolver(f"font-size: {pt_val}pt")

    if 'font-size' in result:
        val = result['font-size']
        match = re.match(r'^(\d+(?:\.\d+)?)pt$', val)
        assert match, f"Expected pt value, got {val}"

        trailing_zeros_match = re.search(r'\.(\d*?)(0+)pt$', val)
        if trailing_zeros_match and len(trailing_zeros_match.group(2)) > 1:
            assert False, f"PT value has excessive trailing zeros: {val}"


if __name__ == "__main__":
    test_non_integer_pt_values_formatting()