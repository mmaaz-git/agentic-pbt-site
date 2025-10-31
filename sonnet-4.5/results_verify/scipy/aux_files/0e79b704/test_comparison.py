import numpy as np
from scipy import odr

print("Testing various parameter combinations:")

test_cases = [
    (-10.0, 0.5),  # Documentation example
    (0.0, 2.0),     # Bug report case
    (1.0, 1.0),     # Matches default estimate
    (5.0, 0.3),     # Another random case
    (-5.0, -0.5),   # Negative beta1
]

for beta0_true, beta1_true in test_cases:
    print(f"\n=== beta0={beta0_true}, beta1={beta1_true} ===")

    x = np.linspace(0.0, 5.0, 20)
    y = beta0_true + np.exp(beta1_true * x)

    if np.any(~np.isfinite(y)) or np.max(np.abs(y)) > 1e10:
        print("Skipping - values too large/infinite")
        continue

    data = odr.Data(x, y)
    odr_obj = odr.ODR(data, odr.exponential)
    output = odr_obj.run()

    y_fitted = output.beta[0] + np.exp(output.beta[1] * x)
    residuals = y - y_fitted
    ssr = np.sum(residuals**2)

    print(f"  Recovered: beta0={output.beta[0]:.6f}, beta1={output.beta[1]:.6f}")
    print(f"  SSR: {ssr:.6e}")
    print(f"  Success: {'YES' if ssr < 1e-10 else 'NO'}")