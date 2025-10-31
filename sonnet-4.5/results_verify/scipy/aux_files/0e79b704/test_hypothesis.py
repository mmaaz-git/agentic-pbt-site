import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from scipy import odr


@given(
    beta0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    beta1=st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_exponential_model_fitting_property(beta0, beta1):
    x = np.linspace(0.0, 5.0, 20)
    y = beta0 + np.exp(beta1 * x)

    assume(np.all(np.isfinite(y)))
    assume(np.max(np.abs(y)) < 1e10)

    data = odr.Data(x, y)

    odr_obj_with_init = odr.ODR(data, odr.exponential, beta0=[beta0, beta1])
    output_with_init = odr_obj_with_init.run()

    y_fitted_with_init = output_with_init.beta[0] + np.exp(output_with_init.beta[1] * x)
    residuals_with_init = y - y_fitted_with_init
    ssr_with_init = np.sum(residuals_with_init**2)

    assert ssr_with_init < 1e-10

    odr_obj_without_init = odr.ODR(data, odr.exponential)
    output_without_init = odr_obj_without_init.run()

    y_fitted_without_init = output_without_init.beta[0] + np.exp(output_without_init.beta[1] * x)
    residuals_without_init = y - y_fitted_without_init
    ssr_without_init = np.sum(residuals_without_init**2)

    assert ssr_without_init < 1e-10

if __name__ == "__main__":
    # Test the specific failing case
    print("Testing with beta0=0.0, beta1=2.0")
    test_exponential_model_fitting_property(0.0, 2.0)
    print("Test passed!")