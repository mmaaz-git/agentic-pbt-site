import numpy as np
from scipy import odr

x = np.linspace(0.0, 5.0, 20)
y_true = 0.0 + np.exp(2.0 * x)

data = odr.Data(x, y_true)
odr_obj = odr.ODR(data, odr.exponential)
output = odr_obj.run()

print(f"True parameters: beta0=0.0, beta1=2.0")
print(f"Recovered: beta0={output.beta[0]}, beta1={output.beta[1]}")

y_fitted = output.beta[0] + np.exp(output.beta[1] * x)
residuals = y_true - y_fitted
ssr = np.sum(residuals**2)

print(f"Sum of squared residuals: {ssr}")