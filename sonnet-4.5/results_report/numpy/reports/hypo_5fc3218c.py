from hypothesis import assume, given, settings, strategies as st
import numpy.f2py.symbolic as sym

simple_expr_chars = st.sampled_from(['x', 'y', 'z', 'a', 'b', 'c'])

@st.composite
def simple_fortran_expr(draw):
    choice = draw(st.integers(min_value=0, max_value=4))
    if choice == 0:
        return draw(simple_expr_chars)
    elif choice == 1:
        return str(draw(st.integers(min_value=-100, max_value=100)))
    elif choice == 2:
        left = draw(simple_fortran_expr())
        right = draw(simple_fortran_expr())
        op = draw(st.sampled_from(['+', '-', '*']))
        return f'({left} {op} {right})'
    elif choice == 3:
        left = draw(simple_fortran_expr())
        right = draw(simple_fortran_expr())
        return f'({left}) / ({right})'
    else:
        base = draw(simple_expr_chars)
        exp = draw(st.integers(min_value=1, max_value=5))
        return f'{base}**{exp}'

@given(simple_fortran_expr())
@settings(max_examples=500)
def test_fromstring_tostring_roundtrip(expr_str):
    try:
        e1 = sym.fromstring(expr_str)
        s = e1.tostring()
        e2 = sym.fromstring(s)
        assert e1 == e2, f"Round-trip failed: {expr_str} -> {s} -> {e2.tostring()}"
    except Exception as ex:
        if "division by zero" in str(ex).lower() or "zerodivision" in str(ex).lower():
            assume(False)
        raise

if __name__ == "__main__":
    test_fromstring_tostring_roundtrip()