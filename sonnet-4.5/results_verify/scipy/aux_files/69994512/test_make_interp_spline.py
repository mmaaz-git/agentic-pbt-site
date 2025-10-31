import numpy as np
import scipy.interpolate as si

# Test make_interp_spline with the same problematic inputs
print("Testing make_interp_spline with close x values:\n")

# Test case 1: x values with 1e-50 gap
x1 = np.array([0.0, 1.0, 1.0 + 1e-50, 2.0])
y1 = np.array([0.0, 1.0, 0.5, 0.0])

print(f"Test 1: x = {x1}")
print(f"        y = {y1}")
try:
    spl = si.make_interp_spline(x1, y1, k=3)
    y_evaluated = spl(x1)
    print(f"Success: y_evaluated = {y_evaluated}")
    print(f"Max error: {np.max(np.abs(y1 - y_evaluated)):.6f}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Test case 2: Hypothesis failing input
x2 = np.array([-1.0, 0.0, 1.2403587833207833e-86, 1.0])
y2 = np.array([0.0, 0.0, 1.0, 0.0])

print(f"Test 2: x = {x2}")
print(f"        y = {y2}")
try:
    spl = si.make_interp_spline(x2, y2, k=3)
    y_evaluated = spl(x2)
    print(f"Success: y_evaluated = {y_evaluated}")
    print(f"Max error: {np.max(np.abs(y2 - y_evaluated)):.6f}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Test with actual duplicates
x3 = np.array([0.0, 1.0, 1.0, 2.0])
y3 = np.array([0.0, 1.0, 0.5, 0.0])

print(f"Test 3 (actual duplicates): x = {x3}")
print(f"                             y = {y3}")
try:
    spl = si.make_interp_spline(x3, y3, k=3)
    y_evaluated = spl(x3)
    print(f"Success: y_evaluated = {y_evaluated}")
    print(f"Max error: {np.max(np.abs(y3 - y_evaluated)):.6f}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Test splrep with actual duplicates for comparison
print("Testing splrep with actual duplicates:")
x4 = np.array([0.0, 1.0, 1.0, 2.0])
y4 = np.array([0.0, 1.0, 0.5, 0.0])

print(f"x = {x4}")
print(f"y = {y4}")
try:
    tck = si.splrep(x4, y4, s=0)
    y_evaluated = si.splev(x4, tck)
    print(f"Success: y_evaluated = {y_evaluated}")
    print(f"Max error: {np.max(np.abs(y4 - y_evaluated)):.6f}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")