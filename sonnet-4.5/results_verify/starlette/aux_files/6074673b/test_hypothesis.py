import numpy as np
import numpy.random
from hypothesis import given, settings, strategies as st


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=500)
def test_negative_binomial_p_zero_returns_garbage(n):
    result = numpy.random.negative_binomial(n, p=0, size=10)

    assert np.all(result == np.iinfo(np.int64).min), \
        f"negative_binomial(n={n}, p=0) returns INT64_MIN instead of raising error or valid value"

if __name__ == "__main__":
    test_negative_binomial_p_zero_returns_garbage()
    print("Test passed - negative_binomial returns INT64_MIN for p=0")