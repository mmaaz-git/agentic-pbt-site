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
        assert result[i] == expected, f"Failed for arr[{i}]={repr(arr[i])}, sub={repr(sub)}, start={start}, end={end}: got {result[i]}, expected {expected}"

# Test with the specific failing input
print("Testing specific failing input from bug report:")
arr = np.array(['abc'], dtype=str)
sub = '\x00'
result = nps.count(arr, sub)
expected = arr[0].count(sub)
print(f"arr = {repr(arr)}, sub = {repr(sub)}")
print(f"NumPy result: {result[0]}")
print(f"Python expected: {expected}")
print(f"Match: {result[0] == expected}")
print()

# Run property-based test
print("Running property-based test...")
try:
    test_count_with_bounds()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")