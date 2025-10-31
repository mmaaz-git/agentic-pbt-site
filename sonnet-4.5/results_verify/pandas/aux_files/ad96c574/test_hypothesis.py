import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.integers(min_value=1, max_value=20))
@settings(max_examples=1000)
def test_slice_with_step(arr, step):
    result = nps.slice(arr, None, None, step)
    for i in range(len(arr)):
        expected = arr[i][::step]
        assert result[i] == expected, f"Failed for arr[{i}]={repr(arr[i])}, step={step}: expected {repr(expected)}, got {repr(result[i])}"

# Run the test
if __name__ == "__main__":
    # First test the specific failing input mentioned
    print("Testing specific failing input from bug report...")
    arr = np.array(['\x000'], dtype=str)
    step = 2
    result = nps.slice(arr, None, None, step)
    expected = '\x000'[::2]
    print(f"Input: arr={repr(arr[0])}, step={step}")
    print(f"Expected: {repr(expected)}")
    print(f"Got: {repr(result[0])}")
    print(f"Match: {result[0] == expected}")
    print()

    # Now run the hypothesis test
    print("Running hypothesis test...")
    try:
        test_slice_with_step()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")