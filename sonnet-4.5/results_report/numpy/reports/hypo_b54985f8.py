import numpy.char as char
from hypothesis import given, strategies as st, settings, assume

@given(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=20))
@settings(max_examples=1000)
def test_replace_respects_pattern_length(s, old):
    assume(len(old) > len(s))

    new = 'REPLACEMENT'

    py_result = s.replace(old, new)
    np_result = str(char.replace(s, old, new))

    if py_result != np_result:
        raise AssertionError(
            f"replace({repr(s)}, {repr(old)}, {repr(new)}): "
            f"Python={repr(py_result)}, NumPy={repr(np_result)}"
        )

if __name__ == "__main__":
    test_replace_respects_pattern_length()