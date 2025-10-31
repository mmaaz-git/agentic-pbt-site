import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, assume


@settings(max_examples=1000)
@given(st.lists(st.text(min_size=0, max_size=30), min_size=1, max_size=10), st.lists(st.text(min_size=0, max_size=30), min_size=1, max_size=10))
def test_add_matches_python(strings1, strings2):
    assume(len(strings1) == len(strings2))
    arr1 = np.array(strings1, dtype='<U100')
    arr2 = np.array(strings2, dtype='<U100')
    result = nps.add(arr1, arr2)

    for i in range(len(strings1)):
        expected = strings1[i] + strings2[i]
        assert result[i] == expected, f"Failed at index {i}: {repr(result[i])} != {repr(expected)}"

# Run the test
if __name__ == "__main__":
    test_add_matches_python()