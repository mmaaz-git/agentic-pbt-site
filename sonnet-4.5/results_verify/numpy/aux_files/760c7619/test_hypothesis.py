import numpy as np
import numpy.char
from hypothesis import given, strategies as st, assume


@given(st.lists(st.text(), min_size=1), st.text())
def test_rpartition_matches_python(strings, sep):
    assume(len(sep) > 0)
    arr = np.array(strings)
    numpy_result = numpy.char.rpartition(arr, sep)

    for i in range(len(strings)):
        python_result = strings[i].rpartition(sep)
        assert tuple(numpy_result[i]) == python_result

# Test with the failing input
if __name__ == "__main__":
    print("Testing with reported failing input...")
    strings = ['']
    sep = '\x00'
    print(f"strings={strings}, sep={repr(sep)}")

    arr = np.array(strings)
    print(f"Testing Python's str.rpartition...")
    python_result = strings[0].rpartition(sep)
    print(f"Python result: {python_result}")

    print(f"Testing NumPy's char.rpartition...")
    try:
        numpy_result = numpy.char.rpartition(arr, sep)
        print(f"NumPy result: {tuple(numpy_result[0])}")
    except Exception as e:
        print(f"NumPy raised: {type(e).__name__}: {e}")