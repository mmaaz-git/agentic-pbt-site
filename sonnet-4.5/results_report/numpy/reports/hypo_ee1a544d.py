from hypothesis import given, strategies as st, assume
import numpy.rec as rec


@given(
    st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_characters=',')), min_size=1, max_size=5),
    st.lists(st.sampled_from(['i4', 'f8', 'i2']), min_size=1, max_size=5)
)
def test_format_parser_name_count_matches(names, formats):
    assume(len(names) == len(formats))
    assume(len(set(names)) == len(names))

    parser = rec.format_parser(formats, names, [])

    assert len(parser.dtype.names) == len(formats)
    for i, name in enumerate(names):
        assert parser.dtype.names[i] == name  # FAILS when name is '\r', '\n', etc.

if __name__ == "__main__":
    test_format_parser_name_count_matches()