import numpy.rec as rec
from hypothesis import given, strategies as st, settings


@given(st.lists(st.tuples(st.integers(), st.text(alphabet=st.characters(min_codepoint=0, max_codepoint=127)), st.floats(allow_nan=False, allow_infinity=False)), min_size=1, max_size=10))
@settings(max_examples=1000)
def test_fromrecords_preserves_string_data(records):
    result = rec.fromrecords(records, names=['a', 'b', 'c'])

    for i, (a, b, c) in enumerate(records):
        assert result[i].b == b, f"String data lost at index {i}: expected {repr(b)}, got {repr(result[i].b)}"

if __name__ == "__main__":
    # Run the test
    test_fromrecords_preserves_string_data()