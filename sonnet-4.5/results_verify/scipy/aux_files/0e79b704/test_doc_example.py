import numpy as np
from scipy import odr

# This is the exact example from the exponential model documentation
x = np.linspace(0.0, 5.0)
y = -10.0 + np.exp(0.5*x)
data = odr.Data(x, y)
odr_obj = odr.ODR(data, odr.exponential)
output = odr_obj.run()
print(f"Beta from documentation example: {output.beta}")
print(f"Expected: [-10.0, 0.5]")

# Check the residuals
y_fitted = output.beta[0] + np.exp(output.beta[1] * x)
residuals = y - y_fitted
ssr = np.sum(residuals**2)
print(f"Sum of squared residuals: {ssr}")