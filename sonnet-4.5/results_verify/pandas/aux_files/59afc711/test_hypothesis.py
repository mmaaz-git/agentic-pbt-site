import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=500, deadline=None)
def test_replace_matches_python(s):
    for old, new in [(s, s + 'x'), (s[0], s[0] * 2)]:
        numpy_result = char.replace(s, old, new)
        numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)
        python_result = s.replace(old, new)
        assert numpy_str == python_result, f"Mismatch: s={repr(s)}, old={repr(old)}, new={repr(new)}, numpy={repr(numpy_str)}, python={repr(python_result)}"

# Run the test
test_replace_matches_python()