from scipy.spatial.distance import correlation
import numpy as np

# Test with the specific failing input
u = [5.0, 5.0, 5.0]
print(f"Testing with constant array: {u}")
u_arr = np.array(u)
d = correlation(u_arr, u_arr)
print(f"correlation({u}, {u}) = {d}")
print(f"Is NaN? {np.isnan(d)}")

# Test the assertion
try:
    assert d == 0.0, f"Expected 0.0 for identical arrays, got {d}"
    print("Assertion passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")