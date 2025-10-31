from hypothesis import given, strategies as st, settings
from numpy.f2py.symbolic import as_string, normalize


@st.composite
def expr_strings(draw):
    s = draw(st.text(min_size=0, max_size=20))
    quote_char = draw(st.sampled_from(['"', "'"]))
    quoted = quote_char + s + quote_char
    return as_string(quoted, kind=1)


@given(expr_strings(), expr_strings(), expr_strings())
@settings(max_examples=500)
def test_string_concat_associative(a, b, c):
    result1 = (a // b) // c
    result2 = a // (b // c)
    assert normalize(result1) == normalize(result2)


if __name__ == "__main__":
    test_string_concat_associative()