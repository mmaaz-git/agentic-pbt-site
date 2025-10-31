import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, example

@given(st.lists(st.text(), min_size=1, max_size=10).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@example(np.array(['abc'], dtype=str), '\x00', 0, None)  # Add the failing example
@settings(max_examples=10)
def test_count_with_bounds(arr, sub, start, end):
    result = nps.count(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].count(sub, start, end)
        assert result[i] == expected, f"Failed for arr[{i}]={repr(arr[i])}, sub={repr(sub)}, start={start}, end={end}: got {result[i]}, expected {expected}"

print("Running property-based test with null character example...")
try:
    test_count_with_bounds()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")