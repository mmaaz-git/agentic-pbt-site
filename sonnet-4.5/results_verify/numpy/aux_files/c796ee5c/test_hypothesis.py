import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=0, max_size=50), min_size=1, max_size=10))
@settings(max_examples=500)
def test_find_matches_python_str(strings):
    arr = np.array(strings)
    numpy_results = char.find(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.find('\x00')
        assert numpy_results[i] == python_result, f"find mismatch for {s!r}: numpy={numpy_results[i]}, python={python_result}"

if __name__ == "__main__":
    test_find_matches_python_str()