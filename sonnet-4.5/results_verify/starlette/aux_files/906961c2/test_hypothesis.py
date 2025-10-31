import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_title_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.title(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].title()

# Test with the failing input specifically
if __name__ == "__main__":
    test_title_unicode()