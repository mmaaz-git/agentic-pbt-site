import scipy.signal.windows as w
import numpy as np

# Test case that demonstrates the NaN issue
result = w.tukey(3, alpha=1e-313, sym=True)
print(f'tukey(3, alpha=1e-313) = {result}')
print(f'Contains NaN: {np.any(np.isnan(result))}')
print()

# Test multiple alpha values to show the boundary
print("Testing different alpha values:")
for alpha in [1e-320, 1e-313, 1e-310, 1e-300, 1e-10]:
    result = w.tukey(5, alpha=alpha)
    has_nan = np.any(np.isnan(result))
    print(f'alpha={alpha:.2e}: has_nan={has_nan}, result={result}')