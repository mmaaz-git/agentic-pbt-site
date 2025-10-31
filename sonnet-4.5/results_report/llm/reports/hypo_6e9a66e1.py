import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import llm


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_cosine_similarity_no_crash_with_zero_vectors(a):
    b = [0.0] * len(a)
    result = llm.cosine_similarity(a, b)
    assert isinstance(result, (int, float))

# Run the test
if __name__ == "__main__":
    test_cosine_similarity_no_crash_with_zero_vectors()