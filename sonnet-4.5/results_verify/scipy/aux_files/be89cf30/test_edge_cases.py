import numpy as np
from scipy.signal import windows

# Test various alpha values
test_cases = [
    (2, 0.0),       # alpha = 0, should be rectangular
    (2, 1e-324),    # Very small but > 0
    (2, 5e-324),    # The reported failing case
    (2, 1e-300),    # Slightly larger but still tiny
    (2, 1e-100),    # Much larger but still very small
    (2, 1e-15),     # Near machine epsilon
    (2, 1e-10),     # Small but manageable
    (2, 0.1),       # Normal small value
    (2, 0.5),       # Default value
    (2, 1.0),       # Should be Hann window
]

print("Testing various alpha values:")
print("-" * 60)
for M, alpha in test_cases:
    try:
        w = windows.tukey(M, alpha)
        has_nan = np.any(np.isnan(w))
        has_inf = np.any(np.isinf(w))
        print(f"M={M}, alpha={alpha:.2e}: values={w}, NaN={has_nan}, Inf={has_inf}")
    except Exception as e:
        print(f"M={M}, alpha={alpha:.2e}: ERROR - {e}")

# Test what happens with negative alpha
print("\n\nTesting negative alpha (should be handled as rectangular):")
try:
    w = windows.tukey(2, -0.1)
    print(f"alpha=-0.1: {w}")
except Exception as e:
    print(f"alpha=-0.1: ERROR - {e}")

# Test larger window sizes with tiny alpha
print("\n\nTesting larger M with tiny alpha:")
for M in [10, 100]:
    w = windows.tukey(M, 5e-324)
    has_nan = np.any(np.isnan(w))
    print(f"M={M}, alpha=5e-324: NaN present={has_nan}, first 5 values={w[:5]}")