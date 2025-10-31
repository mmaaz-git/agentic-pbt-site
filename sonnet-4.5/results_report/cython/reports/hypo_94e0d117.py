from hypothesis import given, strategies as st
from Cython.Utils import strip_py2_long_suffix

@given(st.text(min_size=0, max_size=100))
def test_strip_py2_long_suffix_idempotence(s):
    """Test that strip_py2_long_suffix is idempotent - applying it twice gives the same result as applying it once."""
    result1 = strip_py2_long_suffix(s)
    result2 = strip_py2_long_suffix(result1)
    assert result1 == result2, f"Not idempotent for input: {s!r}"

if __name__ == "__main__":
    test_strip_py2_long_suffix_idempotence()