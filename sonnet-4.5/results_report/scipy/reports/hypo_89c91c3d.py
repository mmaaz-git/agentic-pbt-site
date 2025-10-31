from hypothesis import given, strategies as st, settings
from scipy.odr import ODR, Data, Model
import pytest

def linear_func(B, x):
    return B[0] * x + B[1]

@given(st.integers(min_value=0, max_value=9999))
@settings(max_examples=200)
def test_set_iprint_robust_to_manual_iprint(manual_iprint):
    data = Data([1, 2, 3], [2, 4, 6])
    model = Model(linear_func)
    odr = ODR(data, model, beta0=[1, 1])

    odr.iprint = manual_iprint

    odr.set_iprint(final=0)

if __name__ == "__main__":
    test_set_iprint_robust_to_manual_iprint()