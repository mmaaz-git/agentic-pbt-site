from hypothesis import given, strategies as st, seed, settings, Verbosity
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
@settings(max_examples=100, verbosity=Verbosity.verbose)
@seed(0)
def test_cosine_similarity_no_crash(a, b):
    if len(a) != len(b):
        return
    try:
        result = llm.cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0
    except ZeroDivisionError as e:
        print(f"\nFound failing input: a={a}, b={b}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_cosine_similarity_no_crash()