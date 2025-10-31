from hypothesis import given, strategies as st, settings
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1)
)
@settings(max_examples=500)
def test_cosine_similarity_range(a, b):
    result = llm.cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0

if __name__ == "__main__":
    test_cosine_similarity_range()