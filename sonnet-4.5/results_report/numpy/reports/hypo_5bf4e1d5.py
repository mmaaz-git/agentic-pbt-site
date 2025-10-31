from hypothesis import given, strategies as st, settings
from numpy.f2py import symbolic


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=500)
def test_quote_elimination_round_trip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    reconstructed = symbolic.insert_quotes(new_s, mapping)
    assert s == reconstructed


if __name__ == "__main__":
    # Run the test
    test_quote_elimination_round_trip()