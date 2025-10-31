import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_upper_lower_unicode(strings):
    arr = np.array(strings, dtype=str)
    upper_result = char.upper(arr)

    for i in range(len(strings)):
        assert upper_result[i] == strings[i].upper()

if __name__ == "__main__":
    test_upper_lower_unicode()