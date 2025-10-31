import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=1000)
@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20), st.text(min_size=0, max_size=5), st.text(min_size=0, max_size=5))
def test_replace_matches_python(strings, old, new):
    assume(len(old) > 0)
    arr = np.array(strings)
    replaced = nps.replace(arr, old, new)

    for i, s in enumerate(strings):
        expected = s.replace(old, new)
        assert replaced[i] == expected, f"Failed for string={s}, old={old}, new={new}: expected={expected}, got={replaced[i]}"


# Test with the specific failing case
def test_specific_case():
    strings = ['0']
    old = '0'
    new = '00'

    print("Testing specific case:")
    print(f"strings={strings}, old={old}, new={new}")

    arr = np.array(strings)
    print(f"Input array: {arr}, dtype: {arr.dtype}")

    replaced = nps.replace(arr, old, new)
    print(f"Replaced array: {replaced}, dtype: {replaced.dtype}")

    expected = strings[0].replace(old, new)
    print(f"Expected (Python): {expected}")
    print(f"Actual (NumPy): {replaced[0]}")

    assert replaced[0] == expected, f"Expected {expected}, got {replaced[0]}"


if __name__ == "__main__":
    # Run the specific test case
    try:
        test_specific_case()
        print("\nSpecific test case passed!")
    except AssertionError as e:
        print(f"\nSpecific test case failed: {e}")

    # Run hypothesis tests
    print("\nRunning hypothesis tests...")
    try:
        test_replace_matches_python()
        print("Hypothesis tests passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")