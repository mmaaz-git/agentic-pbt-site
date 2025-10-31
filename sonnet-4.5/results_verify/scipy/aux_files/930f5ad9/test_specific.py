import numpy as np
from scipy.interpolate import PPoly

# Test with the specific failing input
print("Testing with specific failing input:")
x_values = [0.0, 1.0]
c_values = [0.0]

x = np.array(sorted(set(x_values)))
k = len(c_values)
c = np.array(c_values).reshape(k, 1)

pp = PPoly(c, x)
roots = pp.roots()

print(f"PPoly created with x={x}, c={c}")
print(f"roots() returned: {roots}")

if len(roots) > 0:
    root_values = pp(roots)
    print(f"pp(roots) = {root_values}")

    # Check if roots contain NaN
    if np.any(np.isnan(roots)):
        print("ERROR: roots contain NaN values!")

    # Check if evaluating at non-NaN roots gives ~0
    valid_roots = roots[~np.isnan(roots)]
    if len(valid_roots) > 0:
        print(f"Valid (non-NaN) roots: {valid_roots}")
        print(f"pp(valid_roots) = {pp(valid_roots)}")