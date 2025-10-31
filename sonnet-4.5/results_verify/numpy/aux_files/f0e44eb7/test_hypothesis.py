import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)))
@settings(max_examples=1000)
def test_comparison_consistency(arr1, arr2):
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            np_less = nps.less(arr1[i:i+1], arr2[i:i+1])[0]
            py_less = arr1[i] < arr2[i]
            assert np_less == py_less

# Test with the specific failing input mentioned
arr1 = np.array(['a'], dtype=str)
arr2 = np.array(['a\x00'], dtype=str)

print(f"Testing with arr1={repr(arr1)}, arr2={repr(arr2)}")
print(f"arr1[0] = {repr(arr1[0])}")
print(f"arr2[0] = {repr(arr2[0])}")

np_less = nps.less(arr1, arr2)[0]
py_less = arr1[0] < arr2[0]
print(f"numpy.strings.less result: {np_less}")
print(f"Python < result: {py_less}")
print(f"Match: {np_less == py_less}")

# Run hypothesis test
test_comparison_consistency()