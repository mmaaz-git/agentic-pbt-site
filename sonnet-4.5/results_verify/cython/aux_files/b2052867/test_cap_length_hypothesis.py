import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from Cython.Compiler.PyrexTypes import cap_length

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1),
       st.integers(min_value=1, max_value=200))
def test_result_length_bounded(s, max_len):
    result = cap_length(s, max_len)
    if len(s) > max_len:
        assert len(result) <= max_len, f"cap_length('{s}', {max_len}) = '{result}', length={len(result)} > {max_len}"

# Run the test
if __name__ == "__main__":
    try:
        test_result_length_bounded()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")