from hypothesis import given, strategies as st, settings
import pandas.util.version as pv

def test_negative_infinity_equals_itself():
    neg_inf = pv.NegativeInfinity
    assert neg_inf == neg_inf
    assert not (neg_inf != neg_inf)
    assert not (neg_inf < neg_inf)
    assert not (neg_inf > neg_inf)
    assert neg_inf <= neg_inf
    assert neg_inf >= neg_inf

# Run the test
test_negative_infinity_equals_itself()
print("Test passed!")