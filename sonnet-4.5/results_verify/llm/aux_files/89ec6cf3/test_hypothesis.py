import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st, assume, example
import llm

@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10)
)
@example([1, 0], [1])  # The specific example from the bug report
def test_cosine_similarity_length_check(a, b):
    if len(a) != len(b):
        try:
            result = llm.cosine_similarity(a, b)
            assert False, (
                f"cosine_similarity should reject mismatched lengths but returned {result} "
                f"for a (len={len(a)}) and b (len={len(b)})"
            )
        except (ValueError, AssertionError):
            pass

print("Running hypothesis test...")
test_cosine_similarity_length_check()
print("Test completed - if no assertion errors, the bug exists")