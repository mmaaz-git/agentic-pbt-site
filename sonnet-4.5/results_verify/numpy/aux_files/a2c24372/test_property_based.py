import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings


@st.composite
def text_ending_with_null(draw):
    prefix = draw(st.text(min_size=1, max_size=10))
    num_nulls = draw(st.integers(min_value=1, max_value=3))
    return prefix + '\x00' * num_nulls


@given(text_ending_with_null(), st.integers(min_value=1, max_value=5))
@settings(max_examples=200)
def test_bug_multiply_strips_trailing_nulls(s, n):
    arr = np.array([s])
    result = char.multiply(arr, n)[0]
    expected = s * n
    assert result == expected, f"Failed for s={repr(s)}, n={n}: got {repr(result)}, expected {repr(expected)}"

# Run the test
if __name__ == "__main__":
    test_bug_multiply_strips_trailing_nulls()
    print("Test completed without assertion errors!")