import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_upper_matches_python(s):
    numpy_result = char.upper(s)
    numpy_str = str(numpy_result) if hasattr(numpy_result, 'item') else numpy_result
    python_result = s.upper()
    assert numpy_str == python_result

if __name__ == "__main__":
    test_upper_matches_python()