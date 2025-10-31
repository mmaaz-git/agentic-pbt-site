from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.char as char

@given(st.text(max_size=50))
@settings(max_examples=500)
def test_case_preserves_length(s):
    upper_result = char.upper(s)
    upper_str = str(upper_result)
    assert len(upper_str) == len(s), f"upper changed length: {len(s)} -> {len(upper_str)} for input {repr(s)}"

# Run the test
if __name__ == "__main__":
    test_case_preserves_length()