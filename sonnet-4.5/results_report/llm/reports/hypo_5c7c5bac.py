from hypothesis import given, strategies as st
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    result = llm.cosine_similarity(a, b)

# Run the test
test_cosine_similarity_no_crash()