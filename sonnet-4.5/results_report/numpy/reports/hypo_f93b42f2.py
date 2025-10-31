import numpy as np
import numpy.strings
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(), min_size=1), st.text(min_size=1), st.text(min_size=1))
@settings(max_examples=1000)
def test_add_associativity(strings, s1, s2):
    arr = np.array(strings)
    left = numpy.strings.add(numpy.strings.add(arr, s1), s2)
    right = numpy.strings.add(arr, s1 + s2)
    assert np.array_equal(left, right), f"Failed for strings={strings!r}, s1={s1!r}, s2={s2!r}"

if __name__ == "__main__":
    test_add_associativity()