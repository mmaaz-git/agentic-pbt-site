from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text())
def test_eliminate_insert_quotes_roundtrip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    restored = symbolic.insert_quotes(new_s, mapping)
    assert restored == s

# Run the test
if __name__ == "__main__":
    test_eliminate_insert_quotes_roundtrip()