import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, example

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
@example(['\x00'])  # Explicitly add the failing case
@settings(max_examples=500)
def test_multiply_broadcast(strings):
    arr = np.array(strings, dtype=str)
    n = 3
    result = nps.multiply(arr, n)
    for i in range(len(arr)):
        expected = strings[i] * n
        assert result[i] == expected, f"Failed for string {repr(strings[i])}: expected {repr(expected)}, got {repr(result[i])}"

if __name__ == "__main__":
    try:
        test_multiply_broadcast()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()