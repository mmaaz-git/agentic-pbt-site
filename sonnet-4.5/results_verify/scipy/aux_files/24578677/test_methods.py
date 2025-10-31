import numpy as np
from scipy import optimize
import warnings

def model(x, a, b):
    return a * x + b

scale = 1.015625
offset = 0.0

xdata = np.linspace(-10, 10, 50)
ydata = model(xdata, scale, offset)

methods = ['lm', 'trf', 'dogbox']

for method in methods:
    print(f"\nTesting method: {method}")
    print("-" * 40)

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if method == 'lm':
                popt, pcov = optimize.curve_fit(model, xdata, ydata, method=method)
            else:
                # trf and dogbox require bounds or will use lm
                popt, pcov = optimize.curve_fit(model, xdata, ydata, method=method,
                                                bounds=([-np.inf, -np.inf], [np.inf, np.inf]))

            if w:
                print(f"Warning: {w[-1].message}")

            print(f"Parameters: {popt}")
            print(f"Covariance matrix:\n{pcov}")
            print(f"Contains inf: {np.any(np.isinf(pcov))}")
            print(f"Covariance matrix condition number: {np.linalg.cond(pcov) if not np.any(np.isinf(pcov)) else 'inf'}")
    except Exception as e:
        print(f"Error: {e}")