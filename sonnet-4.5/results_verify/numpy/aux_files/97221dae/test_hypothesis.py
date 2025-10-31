import numpy as np
import numpy.strings as nps
from hypothesis import assume, given, strategies as st

@given(st.text(min_size=1), st.integers(min_value=0, max_value=20))
def test_slice_with_explicit_stop(s, start):
    assume(start < len(s))
    arr = np.array([s])
    result = nps.slice(arr, start, None)[0]
    expected = s[start:None]
    print(f"Testing s='{s}', start={start}")
    print(f"Expected: {repr(expected)}")
    print(f"Got: {repr(str(result))}")
    assert str(result) == expected

# Test the specific failing case manually
def test_manual():
    s = 'hello'
    start = 0
    arr = np.array([s])
    result = nps.slice(arr, start, None)[0]
    expected = s[start:None]
    print(f"Testing s='{s}', start={start}")
    print(f"Expected: {repr(expected)}")
    print(f"Got: {repr(str(result))}")
    assert str(result) == expected

test_manual()