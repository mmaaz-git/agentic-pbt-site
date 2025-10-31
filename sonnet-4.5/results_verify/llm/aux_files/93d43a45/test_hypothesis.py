import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from hypothesis import assume, given, strategies as st
import llm

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1))
def test_cosine_similarity_self(a):
    assume(any(x != 0 for x in a))
    result = llm.cosine_similarity(a, a)
    assert result == 1.0

# Run the test
test_cosine_similarity_self()