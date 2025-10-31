from hypothesis import given, strategies as st
from Cython.Compiler.PyrexTypes import cap_length


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)), st.integers(min_value=10, max_value=200))
def test_cap_length_respects_max(s, max_len):
    result = cap_length(s, max_len)
    assert len(result) <= max_len

if __name__ == "__main__":
    test_cap_length_respects_max()