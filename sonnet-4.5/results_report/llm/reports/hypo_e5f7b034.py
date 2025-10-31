import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    try:
        result = llm.cosine_similarity(a, b)
        if sum(x * x for x in a) > 0 and sum(x * x for x in b) > 0:
            assert -1 <= result <= 1
    except ZeroDivisionError:
        assert False, "Should handle zero vectors gracefully"

if __name__ == "__main__":
    test_cosine_similarity_handles_zero_vectors()