from hypothesis import given, strategies as st
import numpy as np
import scipy.odr as odr

@given(n=st.integers(min_value=5, max_value=30))
def test_unilinear_model(n):
    """Property: Built-in unilinear model should work with single parameter"""
    x = np.linspace(1, 10, n)
    y = 2.5 * x

    unilin_model = odr.unilinear
    data = odr.Data(x, y)

    odr_obj = odr.ODR(data, unilin_model, beta0=[1.0])
    output = odr_obj.run()

    assert len(output.beta) == 1
    assert np.isclose(output.beta[0], 2.5, rtol=1e-5)

if __name__ == "__main__":
    test_unilinear_model()