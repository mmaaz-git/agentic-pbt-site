import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


st_text = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00'),
    min_size=0,
    max_size=20
)


@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=1000)
def test_title_matches_python(arr):
    result = char.title(arr)
    for i in range(len(arr)):
        numpy_result = result[i]
        python_result = arr[i].title()
        assert numpy_result == python_result, f"Mismatch at index {i}: numpy={numpy_result!r}, python={python_result!r}, input={arr[i]!r}"

if __name__ == "__main__":
    test_title_matches_python()