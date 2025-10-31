import numpy as np
import scipy.interpolate as si

# Test case from the bug report
x = np.array([0.0, 1.0, 1.0 + 1e-50, 2.0])
y = np.array([0.0, 1.0, 0.5, 0.0])

print("Testing with x values very close together (1e-50 gap):")
print(f"x = {x}")
print(f"y = {y}")

tck = si.splrep(x, y, s=0)
y_evaluated = si.splev(x, tck)

print(f"\nExpected: {y}")
print(f"Got:      {y_evaluated}")
print(f"Max error: {np.max(np.abs(y - y_evaluated)):.6f}")

# Test with different gaps
print("\n\nTesting with different gaps:")
gaps = [1e-10, 1e-15, 1e-20, 1e-30, 1e-50, 1e-100]
for gap in gaps:
    x_test = np.array([0.0, 1.0, 1.0 + gap, 2.0])
    y_test = np.array([0.0, 1.0, 0.5, 0.0])

    try:
        tck_test = si.splrep(x_test, y_test, s=0)
        y_test_evaluated = si.splev(x_test, tck_test)
        max_error = np.max(np.abs(y_test - y_test_evaluated))
        print(f"Gap {gap:1.0e}: Max error = {max_error:.6f}")
    except Exception as e:
        print(f"Gap {gap:1.0e}: Error - {e}")

# Test the failing input from Hypothesis
print("\n\nTesting Hypothesis failing input:")
x_hyp = np.array([-1.0, 0.0, 1.2403587833207833e-86, 1.0])
y_hyp = np.array([0.0, 0.0, 1.0, 0.0])

print(f"x = {x_hyp}")
print(f"y = {y_hyp}")

tck_hyp = si.splrep(x_hyp, y_hyp, s=0)
y_hyp_evaluated = si.splev(x_hyp, tck_hyp)

print(f"Expected: {y_hyp}")
print(f"Got:      {y_hyp_evaluated}")
print(f"Max error: {np.max(np.abs(y_hyp - y_hyp_evaluated)):.6f}")