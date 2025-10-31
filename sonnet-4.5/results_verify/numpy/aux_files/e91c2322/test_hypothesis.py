import numpy.char as char
from hypothesis import given, strategies as st, settings


safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=('Cs',),
        blacklist_characters=' \t\n\r\x00\x0b\x0c'
    ),
    min_size=1
)


@given(safe_text)
@settings(max_examples=1000)
def test_upper_lower_roundtrip(s):
    arr = char.array([s])
    result = char.lower(char.upper(arr))
    expected = s.upper().lower()
    assert result[0] == expected, f"Failed for {repr(s)}: got {repr(result[0])}, expected {repr(expected)}"

if __name__ == "__main__":
    test_upper_lower_roundtrip()