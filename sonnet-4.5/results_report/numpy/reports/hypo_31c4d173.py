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