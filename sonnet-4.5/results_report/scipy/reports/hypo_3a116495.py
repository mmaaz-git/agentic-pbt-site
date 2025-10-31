from hypothesis import given, settings, strategies as st
import scipy.special
import numpy as np

@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_expit_logit_round_trip(x):
    result = scipy.special.logit(scipy.special.expit(x))
    assert np.isclose(result, x, rtol=1e-9, atol=1e-9), \
        f"logit(expit({x})) = {result}, expected {x}"

if __name__ == "__main__":
    test_expit_logit_round_trip()