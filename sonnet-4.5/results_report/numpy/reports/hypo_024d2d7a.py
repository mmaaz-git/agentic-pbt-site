import numpy as np
import numpy.strings as nps
from hypothesis import assume, given, strategies as st

@given(st.text(min_size=1), st.integers(min_value=0, max_value=20))
def test_slice_with_explicit_stop(s, start):
    assume(start < len(s))
    arr = np.array([s])
    result = nps.slice(arr, start, None)[0]
    expected = s[start:None]
    assert str(result) == expected, f"For s={repr(s)}, start={start}: expected {repr(expected)}, got {repr(str(result))}"

# Run the test
if __name__ == "__main__":
    test_slice_with_explicit_stop()