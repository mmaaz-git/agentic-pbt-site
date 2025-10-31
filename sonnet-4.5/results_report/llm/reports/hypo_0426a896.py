import llm
from hypothesis import given, strategies as st
import math

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=10))
def test_cosine_similarity_handles_zero_vectors(b):
    a = [0.0] * len(b)

    try:
        result = llm.cosine_similarity(a, b)
        assert False, f"cosine_similarity with zero vector returned {result}, should raise ValueError"
    except ZeroDivisionError:
        assert False, "Should raise ValueError, not ZeroDivisionError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_cosine_similarity_handles_zero_vectors()