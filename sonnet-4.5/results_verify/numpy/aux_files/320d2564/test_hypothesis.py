import numpy as np
import numpy.char
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(), min_size=1))
@example(['\r'])
def test_chararray_preserves_data(strings):
    arr = numpy.char.array(strings)

    for i in range(len(strings)):
        assert str(arr[i]) == strings[i], f"Mismatch at index {i}: input={repr(strings[i])}, output={repr(str(arr[i]))}"

if __name__ == "__main__":
    # Test with the reported failing input directly
    print("Testing with reported failing input: ['\r']")
    strings = ['\r']
    try:
        arr = numpy.char.array(strings)
        for i in range(len(strings)):
            assert str(arr[i]) == strings[i], f"Mismatch at index {i}: input={repr(strings[i])}, output={repr(str(arr[i]))}"
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed: {e}")