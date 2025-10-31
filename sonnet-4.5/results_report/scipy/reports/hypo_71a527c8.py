from hypothesis import given, strategies as st
from scipy.constants import precision, find


@given(st.sampled_from(find()))
def test_precision_is_always_nonnegative(key):
    prec = precision(key)
    assert prec >= 0, f"precision('{key}') returned {prec}, which is negative"


if __name__ == "__main__":
    test_precision_is_always_nonnegative()