from hypothesis import given, strategies as st, settings
from Cython.Compiler.PyrexTypes import cap_length


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'), st.integers(min_value=0, max_value=200))
@settings(max_examples=500)
def test_cap_length_honors_max_len(s, max_len):
    result = cap_length(s, max_len)
    assert len(result) <= max_len, f"Result '{result}' has length {len(result)} but max_len was {max_len}"

if __name__ == "__main__":
    test_cap_length_honors_max_len()