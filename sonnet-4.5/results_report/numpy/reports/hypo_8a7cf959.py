import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

string_arrays = st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str))

@given(string_arrays, st.text(min_size=1, max_size=10))
@settings(max_examples=1000)
def test_rfind_consistency(arr, sub):
    result = nps.rfind(arr, sub)
    for i in range(len(arr)):
        expected = arr[i].rfind(sub)
        assert result[i] == expected, f"Failed on arr[{i}]='{arr[i]}', sub='{sub}': expected {expected}, got {result[i]}"

if __name__ == "__main__":
    # Run the test
    try:
        test_rfind_consistency()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nTrying specific failing case:")
        arr = np.array(['abc'], dtype=str)
        sub = '\x00'
        result = nps.rfind(arr, sub)
        expected = arr[0].rfind(sub)
        print(f"arr = np.array(['abc'], dtype=str)")
        print(f"sub = '\\x00'")
        print(f"Python str.rfind: {expected}")
        print(f"NumPy rfind: {result[0]}")
        print(f"Expected: -1, Got: {result[0]}")