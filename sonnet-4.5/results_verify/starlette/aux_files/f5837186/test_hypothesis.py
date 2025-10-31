import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=10))
@settings(max_examples=1000)
def test_startswith_consistency(arr, prefix):
    result = nps.startswith(arr, prefix)
    for i in range(len(arr)):
        expected = arr[i].startswith(prefix)
        assert result[i] == expected, f"Mismatch for arr[{i}]='{arr[i]}', prefix='{prefix}': NumPy={result[i]}, Python={expected}"

# Test with the specific failing input
print("Testing specific failing input from bug report...")
arr = np.array(['abc'], dtype=str)
prefix = '\x00'
result = nps.startswith(arr, prefix)
expected = arr[0].startswith(prefix)
print(f"arr = {arr}")
print(f"prefix = repr({repr(prefix)})")
print(f"NumPy result: {result[0]}")
print(f"Python result: {expected}")
print(f"Match: {result[0] == expected}")

# Run the hypothesis test
print("\nRunning Hypothesis test...")
try:
    test_startswith_consistency()
    print("Hypothesis test passed (no failures found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")