import keyword
from hypothesis import given, strategies as st
import pytest
import Cython.Tempita as tempita

RESERVED = {"if", "for", "endif", "endfor", "else", "elif", "py", "default", "inherit"} | set(keyword.kwlist)
valid_identifier = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s not in RESERVED and s.isidentifier())


@given(valid_identifier, st.integers())
def test_bunch_delattr_not_supported(attr_name, value):
    b = tempita.bunch(**{attr_name: value})

    assert hasattr(b, attr_name)
    assert getattr(b, attr_name) == value

    with pytest.raises(AttributeError):
        delattr(b, attr_name)

if __name__ == "__main__":
    # Test manually with the specific failing input
    attr_name = 'a'
    value = 0
    b = tempita.bunch(**{attr_name: value})

    assert hasattr(b, attr_name)
    assert getattr(b, attr_name) == value

    try:
        delattr(b, attr_name)
        print("ERROR: delattr succeeded when it should have failed")
    except AttributeError as e:
        print(f"Test passed - delattr raised AttributeError as expected: {e}")