from hypothesis import given, strategies as st, settings, example
import numpy.f2py.symbolic as symbolic

@given(st.text())
@example('(')
@settings(max_examples=10)
def test_replace_unreplace_parenthesis_roundtrip(s):
    new_s, mapping = symbolic.replace_parenthesis(s)
    restored = symbolic.unreplace_parenthesis(new_s, mapping)
    assert restored == s

if __name__ == "__main__":
    test_replace_unreplace_parenthesis_roundtrip()