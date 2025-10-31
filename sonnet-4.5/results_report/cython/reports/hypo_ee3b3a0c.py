import keyword
from hypothesis import given, strategies as st
import Cython.Tempita as tempita

RESERVED = {"if", "for", "endif", "endfor", "else", "elif", "py", "default", "inherit"} | set(keyword.kwlist)
valid_identifier = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s not in RESERVED and s.isidentifier())


@given(st.dictionaries(valid_identifier, st.integers(), min_size=1, max_size=5))
def test_bunch_dir_missing_attrs(kwargs):
    b = tempita.bunch(**kwargs)

    for key in kwargs:
        assert hasattr(b, key)
        assert key not in dir(b)


if __name__ == "__main__":
    # Run the test
    test_bunch_dir_missing_attrs()