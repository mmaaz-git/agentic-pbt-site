import numpy as np
import scipy.special

print("For large NEGATIVE x (works perfectly):")
for x in [-10, -20, -30, -40, -50, -100]:
    result = scipy.special.logit(scipy.special.expit(x))
    error = abs(result - x)
    print(f"  logit(expit({x:4})) = {result:8.2f}, error = {error:.2e}")

print("\nFor large POSITIVE x (catastrophic failure):")
for x in [10, 20, 30, 40, 50, 100]:
    result = scipy.special.logit(scipy.special.expit(x))
    if np.isfinite(result):
        error = abs(result - x)
        print(f"  logit(expit({x:4})) = {result:8.2f}, error = {error:.2e}")
    else:
        print(f"  logit(expit({x:4})) = {result:>8}, error = inf")