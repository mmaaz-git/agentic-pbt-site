import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
@settings(max_examples=500)
def test_multiply_broadcast(strings):
    arr = np.array(strings, dtype=str)
    n = 3
    result = nps.multiply(arr, n)
    for i in range(len(arr)):
        expected = strings[i] * n
        assert result[i] == expected, f"Failed for string {repr(strings[i])}: expected {repr(expected)}, got {repr(result[i])}"

# Test with the specific failing case
if __name__ == "__main__":
    try:
        test_multiply_broadcast()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Test the specific failing input
    print("\nTesting specific failing input: ['\x00']")
    strings = ['\x00']
    arr = np.array(strings, dtype=str)
    n = 3
    result = nps.multiply(arr, n)
    expected = strings[0] * n
    print(f"Input: {repr(strings[0])}")
    print(f"Expected: {repr(expected)}")
    print(f"Got: {repr(result[0])}")
    print(f"Match: {result[0] == expected}")