import pandas as pd
import numpy as np

# Simple reproduction from the bug report
print("Simple reproduction example:")
data = [0.0, 0.0, 0.0, 0.0, 7.797e-124]
s = pd.Series(data)

corr = s.rolling(window=2).corr()
print(f"Data: {data}")
print(f"Rolling correlation:")
print(corr)
print()

# Let's also check what happens with the window at index 4
print("Window at index 4: [0.0, 7.797e-124]")
print(f"Variance of window: {np.var([0.0, 7.797e-124])}")
print(f"Standard deviation: {np.std([0.0, 7.797e-124])}")

# Check autocorrelation formula
x = np.array([0.0, 7.797e-124])
mean_x = x.mean()
var_x = ((x - mean_x) ** 2).mean()
std_x = np.sqrt(var_x)
print(f"Mean: {mean_x}")
print(f"Variance: {var_x}")
print(f"Standard deviation: {std_x}")

# Manually compute autocorrelation
n = len(x)
x_shifted = x[:-1]
x_lag = x[1:]
print(f"\nManual autocorrelation calculation:")
print(f"x_shifted: {x_shifted}")
print(f"x_lag: {x_lag}")

# This is essentially what pandas does for autocorrelation
cov = np.mean((x_shifted - x_shifted.mean()) * (x_lag - x_lag.mean()))
std_shifted = np.std(x_shifted)
std_lag = np.std(x_lag)
print(f"Covariance: {cov}")
print(f"Std of shifted: {std_shifted}")
print(f"Std of lag: {std_lag}")
print(f"Product of stds: {std_shifted * std_lag}")

if std_shifted * std_lag == 0:
    print("Division by zero will occur!")
else:
    manual_corr = cov / (std_shifted * std_lag)
    print(f"Manual correlation: {manual_corr}")