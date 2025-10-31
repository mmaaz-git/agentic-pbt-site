from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text(alphabet='()[]{}', min_size=1, max_size=30))
def test_replace_unreplace_parenthesis_roundtrip(s):
    s_no_parens, d = symbolic.replace_parenthesis(s)
    s_restored = symbolic.unreplace_parenthesis(s_no_parens, d)
    assert s == s_restored

if __name__ == "__main__":
    test_replace_unreplace_parenthesis_roundtrip()