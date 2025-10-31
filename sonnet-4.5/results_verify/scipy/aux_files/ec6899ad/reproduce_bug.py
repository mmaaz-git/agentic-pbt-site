import numpy as np
from scipy.signal.windows import tukey

# Test with small alpha values
w = tukey(2, alpha=1e-309, sym=True)
print(f"tukey(2, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

w = tukey(10, alpha=1e-309, sym=True)
print(f"tukey(10, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Let's test with more values to understand the behavior
print("Testing various small alpha values:")
for alpha_exp in [-50, -100, -150, -200, -250, -300, -309]:
    alpha = 10.0 ** alpha_exp
    try:
        w = tukey(5, alpha=alpha, sym=True)
        has_nan = np.any(np.isnan(w))
        print(f"  alpha=1e{alpha_exp}: has_nan={has_nan}, values={w}")
    except Exception as e:
        print(f"  alpha=1e{alpha_exp}: error={e}")