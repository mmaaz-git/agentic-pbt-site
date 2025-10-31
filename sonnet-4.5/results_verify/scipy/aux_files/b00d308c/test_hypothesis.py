from hypothesis import given, settings, strategies as st
import scipy.stats

@given(
    st.integers(min_value=1, max_value=100),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_binom_edge_probabilities(n, p):
    pmf_0 = scipy.stats.binom.pmf(0, n, p)
    pmf_n = scipy.stats.binom.pmf(n, n, p)

    assert pmf_0 >= 0
    assert pmf_n >= 0

    if p == 0:
        assert pmf_0 == 1.0
        assert pmf_n == 0.0
    elif p == 1:
        assert pmf_0 == 0.0
        assert pmf_n == 1.0

if __name__ == "__main__":
    test_binom_edge_probabilities()