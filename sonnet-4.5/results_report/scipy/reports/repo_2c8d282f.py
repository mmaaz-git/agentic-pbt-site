import numpy as np
from scipy import odr

# Create exact exponential data
x = np.linspace(0.0, 5.0, 20)
y_true = 0.0 + np.exp(2.0 * x)

# Fit without initial parameters (relies on _exp_est)
data = odr.Data(x, y_true)
odr_obj = odr.ODR(data, odr.exponential)
output = odr_obj.run()

print(f"True parameters: beta0=0.0, beta1=2.0")
print(f"Recovered without init: beta0={output.beta[0]}, beta1={output.beta[1]}")

# Calculate residuals
y_fitted = output.beta[0] + np.exp(output.beta[1] * x)
residuals = y_true - y_fitted
ssr = np.sum(residuals**2)
print(f"Sum of squared residuals (without init): {ssr}")

# Now fit WITH initial parameters
odr_obj_with_init = odr.ODR(data, odr.exponential, beta0=[0.0, 2.0])
output_with_init = odr_obj_with_init.run()

print(f"\nRecovered with init [0.0, 2.0]: beta0={output_with_init.beta[0]}, beta1={output_with_init.beta[1]}")

y_fitted_with_init = output_with_init.beta[0] + np.exp(output_with_init.beta[1] * x)
residuals_with_init = y_true - y_fitted_with_init
ssr_with_init = np.sum(residuals_with_init**2)
print(f"Sum of squared residuals (with init): {ssr_with_init}")