import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=10).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5), st.integers(min_value=0, max_value=10))
@settings(max_examples=1000)
def test_replace_count_parameter(arr, old, count):
    result = nps.replace(arr, old, 'X', count=count)
    for i in range(len(arr)):
        expected = arr[i].replace(old, 'X', count)
        assert result[i] == expected, f"Failed for arr[{i}]='{arr[i]}', old='{old}', count={count}. Expected: '{expected}', Got: '{result[i]}'"

# Test the specific failing case
print("Testing specific failing input from bug report:")
arr = np.array([''], dtype=str)
old = '\x00'
count = 1
print(f"arr = {repr(arr)}, old = {repr(old)}, count = {count}")
result = nps.replace(arr, old, 'X', count=count)
expected = ''.replace(old, 'X', count)
print(f"Expected: {repr(expected)}")
print(f"Got: {repr(result[0])}")
print(f"Match: {result[0] == expected}")

# Run the hypothesis test
print("\nRunning Hypothesis test...")
try:
    test_replace_count_parameter()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")