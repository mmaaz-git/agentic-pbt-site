import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=10))
@settings(max_examples=1000)
def test_endswith_consistency(arr, suffix):
    result = nps.endswith(arr, suffix)
    for i in range(len(arr)):
        expected = arr[i].endswith(suffix)
        assert result[i] == expected, f"Mismatch for arr[{i}]={repr(arr[i])}, suffix={repr(suffix)}: numpy={result[i]}, python={expected}"

# Test with the specific failing case mentioned
print("Testing specific failing case from bug report:")
arr = np.array(['abc'], dtype=str)
suffix = '\x00'
result = nps.endswith(arr, suffix)
expected = arr[0].endswith(suffix)
print(f"Array: {repr(arr)}")
print(f"Suffix: {repr(suffix)}")
print(f"NumPy result: {result[0]}")
print(f"Python result: {expected}")
print(f"Match: {result[0] == expected}")

print("\n" + "="*60)
print("Running hypothesis test...")
try:
    test_endswith_consistency()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")