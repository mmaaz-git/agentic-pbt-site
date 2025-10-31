import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=5))
@settings(max_examples=500)
def test_add_broadcast(strings):
    arr = np.array(strings, dtype=str)
    scalar = 'test'
    result = nps.add(arr, scalar)
    for i in range(len(arr)):
        expected = strings[i] + scalar
        assert result[i] == expected, f"Failed at index {i}: {repr(result[i])} != {repr(expected)}"

# Run the test
if __name__ == "__main__":
    test_add_broadcast()