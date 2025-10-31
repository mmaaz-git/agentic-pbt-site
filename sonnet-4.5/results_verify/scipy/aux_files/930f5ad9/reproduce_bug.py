import numpy as np
from scipy.interpolate import PPoly

x = np.array([0.0, 1.0])
c = np.array([[0.0]])

pp = PPoly(c, x)
roots = pp.roots()

print(f"Roots: {roots}")
print(f"Contains NaN: {np.any(np.isnan(roots))}")

try:
    assert not np.any(np.isnan(roots)), "PPoly.roots() should never return NaN"
except AssertionError as e:
    print(f"AssertionError: {e}")