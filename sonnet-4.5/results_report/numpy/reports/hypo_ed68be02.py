import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_capitalize_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.capitalize(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].capitalize(), f"Failed on input: {strings[i]!r}, got {result[i]!r}, expected {strings[i].capitalize()!r}"

if __name__ == "__main__":
    test_capitalize_unicode()