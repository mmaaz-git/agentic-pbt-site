import numpy as np
import scipy.interpolate as interp

x = np.array([5e-324, 0.0, 0.5, 1.0, 2.0])
y = np.sin(x)

poly = interp.lagrange(x, y)

for xi, yi in zip(x, y):
    result = poly(xi)
    print(f"poly({xi:.2e}) = {result}, expected {yi:.10f}")