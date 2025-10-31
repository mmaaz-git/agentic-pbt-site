import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=10),
       st.integers(min_value=-1, max_value=10))
@example(strings=['\x00'], count=0)  # The reported failing case
def test_replace_count_parameter(strings, count):
    print(f"Testing: strings={strings}, count={count}")
    arr = np.array(strings)
    old = 'a'
    new = 'b'
    replaced = nps.replace(arr, old, new, count=count)

    for i, (original, result) in enumerate(zip(strings, replaced)):
        if count == -1:
            expected = original.replace(old, new)
        else:
            expected = original.replace(old, new, count)
        print(f"  Original: {repr(original)}, Result: {repr(result)}, Expected: {repr(expected)}")
        assert result == expected, f"Mismatch at index {i}: {repr(result)} != {repr(expected)}"

if __name__ == "__main__":
    # Run just the specific failing example
    print("Running the reported failing case:")
    try:
        test_replace_count_parameter(['\x00'], 0)
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nChecking array storage directly:")
    arr = np.array(['\x00'])
    print(f"Input: {repr(['\x00'])}")
    print(f"Array: {repr(arr)}")
    print(f"Array[0]: {repr(arr[0])}")
    print(f"Array[0] == '\\x00': {arr[0] == '\x00'}")