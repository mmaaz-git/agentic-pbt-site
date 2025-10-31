import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=500, deadline=None)
def test_replace_matches_python(s):
    for old, new in [(s, s + 'x'), (s[0], s[0] * 2)]:
        numpy_result = char.replace(s, old, new)
        numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)
        python_result = s.replace(old, new)
        assert numpy_str == python_result, f"numpy.char.replace({repr(s)}, {repr(old)}, {repr(new)}) returned {repr(numpy_str)}, expected {repr(python_result)}"

if __name__ == "__main__":
    test_replace_matches_python()