from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10), st.text(min_size=1, max_size=5), st.text(max_size=5))
def test_replace_matches_python(strings, old, new):
    for s in strings:
        if old in s:
            arr = np.array([s])
            np_result = nps.replace(arr, old, new)[0]
            py_result = s.replace(old, new)
            assert np_result == py_result, f"Failed for s='{s}', old='{old}', new='{new}': expected '{py_result}' but got '{np_result}'"

if __name__ == "__main__":
    test_replace_matches_python()