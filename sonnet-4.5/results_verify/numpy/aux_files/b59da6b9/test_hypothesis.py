import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=0, max_size=20))
@settings(max_examples=1000)
def test_str_len_matches_python_len(s):
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    assert numpy_len == python_len, f"Failed for string {s!r}: numpy_len={numpy_len}, python_len={python_len}"

if __name__ == "__main__":
    # Run the hypothesis test
    test_str_len_matches_python_len()
    print("Test passed!")