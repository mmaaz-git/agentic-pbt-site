from hypothesis import given, strategies as st, settings
from Cython.Plex.Regexps import Range

@given(st.text(min_size=1).filter(lambda s: len(s) % 2 == 1))
@settings(max_examples=200)
def test_range_validates_even_length(s):
    Range(s)

if __name__ == "__main__":
    test_range_validates_even_length()