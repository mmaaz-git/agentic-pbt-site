import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1))
def test_swapcase_matches_python(strings):
    arr = np.array(strings)
    numpy_result = numpy.char.swapcase(arr)

    for i in range(len(strings)):
        python_result = strings[i].swapcase()
        assert numpy_result[i] == python_result

# Test with the failing input mentioned in the report
if __name__ == "__main__":
    print("Testing with the specific failing input: ['ẖ']")

    # Direct test without hypothesis decorator
    strings = ['ẖ']
    arr = np.array(strings)
    numpy_result = numpy.char.swapcase(arr)

    match = True
    for i in range(len(strings)):
        python_result = strings[i].swapcase()
        if numpy_result[i] != python_result:
            match = False
            print(f"Test failed for '{strings[i]}'")
            print(f"NumPy result: {repr(numpy_result[i])}")
            print(f"Python result: {repr(python_result)}")
            print(f"Match: {numpy_result[i] == python_result}")

    if match:
        print("Test passed!")
    else:
        print("\nTest failed with mismatch")