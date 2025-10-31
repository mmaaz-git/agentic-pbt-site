import numpy.f2py.symbolic as symbolic
from hypothesis import given, strategies as st, settings

st_symbol_name = st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz')

@st.composite
def st_simple_expr(draw):
    choice = draw(st.integers(min_value=0, max_value=2))
    if choice == 0:
        return symbolic.as_number(draw(st.integers(min_value=-1000, max_value=1000)))
    elif choice == 1:
        return symbolic.as_symbol(draw(st_symbol_name))
    else:
        left = symbolic.as_symbol(draw(st_symbol_name))
        right = symbolic.as_symbol(draw(st_symbol_name))
        return left * right

@given(st_simple_expr())
@settings(max_examples=500)
def test_expr_parse_tostring_roundtrip(expr):
    s = expr.tostring()
    parsed = symbolic.Expr.parse(s)
    assert parsed == expr

if __name__ == "__main__":
    test_expr_parse_tostring_roundtrip()