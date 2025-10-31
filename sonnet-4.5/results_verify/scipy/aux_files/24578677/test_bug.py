import numpy as np
from scipy import optimize
import warnings

def model(x, a, b):
    return a * x + b

scale = 1.015625
offset = 0.0

xdata = np.linspace(-10, 10, 50)
ydata = model(xdata, scale, offset)

# Catch warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    popt, pcov = optimize.curve_fit(model, xdata, ydata)

    # Check if warning was issued
    if w:
        print(f"Warning message: {w[-1].message}")
        print(f"Warning category: {w[-1].category}")

print(f"Parameters recovered: {popt}")
print(f"Covariance matrix:\n{pcov}")
print(f"Contains inf: {np.any(np.isinf(pcov))}")
print(f"All inf: {np.all(np.isinf(pcov))}")