from hypothesis import given, strategies as st, settings
import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.10/site-packages')

@settings(max_examples=1000)
@given(st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127), min_size=0, max_size=10))
def test_string_with_trailing_null(prefix):
    s = prefix + '\x00'
    arr = np.array([s], dtype='U50')

    assert arr[0] == s, f"Expected {repr(s)} but got {repr(arr[0])}"

# Run the test
if __name__ == "__main__":
    test_string_with_trailing_null()