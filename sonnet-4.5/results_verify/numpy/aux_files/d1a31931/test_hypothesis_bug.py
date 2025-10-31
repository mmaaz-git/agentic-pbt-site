import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
def test_partition_matches_python(strings, sep):
    arr = np.array(strings)
    numpy_result = numpy.char.partition(arr, sep)

    for i in range(len(strings)):
        python_result = strings[i].partition(sep)
        assert tuple(numpy_result[i]) == python_result, f"Mismatch at index {i}: numpy={tuple(numpy_result[i])}, python={python_result}"


# Test with the specific failing input
def test_specific_failing_case():
    strings = ['\x00']
    sep = '0'

    arr = np.array(strings)
    numpy_result = numpy.char.partition(arr, sep)

    python_result = strings[0].partition(sep)
    numpy_tuple = tuple(numpy_result[0])

    print(f"Python partition: {python_result}")
    print(f"NumPy partition:  {numpy_tuple}")
    print(f"Match: {numpy_tuple == python_result}")

    assert numpy_tuple == python_result, f"Mismatch: numpy={numpy_tuple}, python={python_result}"


if __name__ == "__main__":
    # Run the specific failing case
    print("Testing specific failing case...")
    try:
        test_specific_failing_case()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run hypothesis test
    print("\nRunning Hypothesis test...")
    try:
        test_partition_matches_python()
    except Exception as e:
        print(f"Hypothesis test failed: {e}")