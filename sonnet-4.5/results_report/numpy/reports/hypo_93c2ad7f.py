import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=10).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@settings(max_examples=1000)
def test_count_with_bounds(arr, sub, start, end):
    result = nps.count(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].count(sub, start, end)
        assert result[i] == expected, f"Failed for arr[{i}]={repr(arr[i])}, sub={repr(sub)}, start={start}, end={end}: expected {expected}, got {result[i]}"

# Run the test
if __name__ == "__main__":
    # Test with the specific failing example
    arr = np.array(['abc'], dtype=str)
    sub = '\x00'
    start = 0
    end = None

    result = nps.count(arr, sub, start, end)
    expected = arr[0].count(sub, start, end)
    print(f"Testing: arr={repr(arr[0])}, sub={repr(sub)}")
    print(f"Expected: {expected}")
    print(f"Got: {result[0]}")

    if result[0] != expected:
        print(f"\nAssertion Error: Failed for arr[0]={repr(arr[0])}, sub={repr(sub)}, start={start}, end={end}: expected {expected}, got {result[0]}")

    # Run the property-based test
    try:
        test_count_with_bounds()
        print("\nProperty-based test passed!")
    except AssertionError as e:
        print(f"\nProperty-based test failed with: {e}")