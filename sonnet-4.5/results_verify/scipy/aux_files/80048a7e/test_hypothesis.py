from hypothesis import given, strategies as st
import numpy as np
import scipy.odr as odr

@given(n=st.integers(min_value=5, max_value=30))
def test_delta0_initialization(n):
    """Property: delta0 can be provided for initialization"""
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.RandomState(42).randn(n) * 0.1

    def linear_func(B, x):
        return B[0] * x + B[1]

    model = odr.Model(linear_func)
    data = odr.Data(x, y)

    delta0 = np.zeros(n)

    odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0)
    output = odr_obj.run()

    assert hasattr(output, 'delta')

# Run the test
test_delta0_initialization()