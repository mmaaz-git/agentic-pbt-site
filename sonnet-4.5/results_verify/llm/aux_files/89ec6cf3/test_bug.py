from hypothesis import given, settings, strategies as st, assume
import llm

@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10)
)
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

if __name__ == "__main__":
    test_cosine_similarity_length_check()
    print("Test completed")