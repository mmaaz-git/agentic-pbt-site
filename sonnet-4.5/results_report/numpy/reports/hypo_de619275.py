import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, example

@given(st.lists(st.text(min_size=1), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=10))
@example(arr=np.array(['abc'], dtype=str), suffix='\x00')
@settings(max_examples=1000)
def test_endswith_consistency(arr, suffix):
    """
    Test that numpy.strings.endswith() returns the same results
    as Python's str.endswith() method.
    """
    result = nps.endswith(arr, suffix)
    for i in range(len(arr)):
        expected = arr[i].endswith(suffix)
        assert result[i] == expected, f"Mismatch for {repr(arr[i])} with suffix {repr(suffix)}: NumPy={result[i]}, Python={expected}"

if __name__ == "__main__":
    print("Running Hypothesis property-based test for numpy.strings.endswith()")
    print("=" * 70)
    try:
        test_endswith_consistency()
        print("All tests passed!")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        print("\nThis demonstrates that numpy.strings.endswith() has incorrect behavior")
        print("when checking for null character '\\x00' as a suffix.")