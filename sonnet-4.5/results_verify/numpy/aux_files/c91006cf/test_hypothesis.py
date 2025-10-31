import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, example

string_arrays = st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str))

@given(string_arrays, st.text(min_size=1, max_size=10))
@example(np.array(['abc'], dtype=str), '\x00')  # The specific failing case
@settings(max_examples=100)
def test_rfind_consistency(arr, sub):
    """Test that numpy.strings.rfind matches Python's str.rfind behavior"""
    result = nps.rfind(arr, sub)
    for i in range(len(arr)):
        expected = arr[i].rfind(sub)
        actual = result[i]
        if actual != expected:
            print(f"\nFAILURE:")
            print(f"  String: {repr(arr[i])}")
            print(f"  Substring: {repr(sub)}")
            print(f"  Expected (Python): {expected}")
            print(f"  Actual (NumPy): {actual}")
            assert False, f"Mismatch for string {repr(arr[i])} finding {repr(sub)}: expected {expected}, got {actual}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_rfind_consistency()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")