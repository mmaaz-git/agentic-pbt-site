import numpy as np
from scipy import odr

def test_specific_case():
    beta0 = 0.0
    beta1 = 2.0

    x = np.linspace(0.0, 5.0, 20)
    y = beta0 + np.exp(beta1 * x)

    data = odr.Data(x, y)

    # Test with initial parameters
    print("\n=== Testing WITH initial parameters ===")
    odr_obj_with_init = odr.ODR(data, odr.exponential, beta0=[beta0, beta1])
    output_with_init = odr_obj_with_init.run()

    y_fitted_with_init = output_with_init.beta[0] + np.exp(output_with_init.beta[1] * x)
    residuals_with_init = y - y_fitted_with_init
    ssr_with_init = np.sum(residuals_with_init**2)

    print(f"Recovered parameters: beta0={output_with_init.beta[0]}, beta1={output_with_init.beta[1]}")
    print(f"Sum of squared residuals: {ssr_with_init}")
    print(f"SSR < 1e-10? {ssr_with_init < 1e-10}")

    # Test without initial parameters
    print("\n=== Testing WITHOUT initial parameters ===")
    odr_obj_without_init = odr.ODR(data, odr.exponential)
    output_without_init = odr_obj_without_init.run()

    y_fitted_without_init = output_without_init.beta[0] + np.exp(output_without_init.beta[1] * x)
    residuals_without_init = y - y_fitted_without_init
    ssr_without_init = np.sum(residuals_without_init**2)

    print(f"Recovered parameters: beta0={output_without_init.beta[0]}, beta1={output_without_init.beta[1]}")
    print(f"Sum of squared residuals: {ssr_without_init}")
    print(f"SSR < 1e-10? {ssr_without_init < 1e-10}")

if __name__ == "__main__":
    test_specific_case()