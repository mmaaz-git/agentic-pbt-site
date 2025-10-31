import numpy as np
import numpy.char
from hypothesis import given, strategies as st

@given(st.lists(st.text(), min_size=1), st.integers(min_value=0, max_value=100))
def test_multiply_length_property(strings, n):
    arr = np.array(strings)
    result = numpy.char.multiply(arr, n)
    for i, s in enumerate(strings):
        assert len(result[i]) == len(s) * n, f"Failed for string {s!r} with n={n}"

# Test with the specific failing input
def test_specific_case():
    strings = ['\x00']
    n = 1
    arr = np.array(strings)
    result = numpy.char.multiply(arr, n)
    print(f"Input string: {strings[0]!r}, length: {len(strings[0])}")
    print(f"Result string: {result[0]!r}, length: {len(result[0])}")
    print(f"Expected length: {len(strings[0]) * n}")
    print(f"Actual length: {len(result[0])}")
    assert len(result[0]) == len(strings[0]) * n

if __name__ == "__main__":
    # Test the specific failing case
    try:
        test_specific_case()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run the hypothesis test
    print("\nRunning hypothesis test...")
    try:
        test_multiply_length_property()
    except Exception as e:
        print(f"Hypothesis test found failure: {e}")