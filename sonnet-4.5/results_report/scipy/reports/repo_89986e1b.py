import numpy as np
import scipy.signal.windows as windows
import warnings

# Show all warnings
warnings.filterwarnings('error')

print("Testing scipy.signal.windows.tukey with extremely small alpha values")
print("=" * 70)

# Test case 1: The specific failing input from hypothesis
print("\nTest 1: Original failing input from hypothesis")
print("-" * 50)
M, alpha = 2, 2.225e-311
try:
    w = windows.tukey(M, alpha)
    print(f"tukey({M}, alpha={alpha:.3e}) = {w}")
    if np.any(np.isnan(w)):
        print("ERROR: NaN values found in output!")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test case 2: Various small alpha values
print("\nTest 2: Various extremely small alpha values")
print("-" * 50)
test_cases = [
    (2, 1e-308),
    (5, 1e-308),
    (10, 1e-308),
    (3, 1e-307),
    (4, 5e-308),
]

for M, alpha in test_cases:
    try:
        warnings.filterwarnings('always')  # Show warnings but don't raise
        with warnings.catch_warnings(record=True) as w_list:
            w = windows.tukey(M, alpha)
            print(f"tukey({M}, alpha={alpha:.2e}) = {w}")
            if np.any(np.isnan(w)):
                print(f"  -> Contains NaN at indices: {np.where(np.isnan(w))[0]}")
            if w_list:
                for warning in w_list:
                    print(f"  -> Warning: {warning.message}")
    except Exception as e:
        print(f"  -> Exception: {type(e).__name__}: {e}")

# Test case 3: Find the threshold where NaN starts appearing
print("\nTest 3: Finding the threshold where NaN appears")
print("-" * 50)
M = 5
alphas = [1e-306, 1e-307, 1e-308, 1e-309, 1e-310]
for alpha in alphas:
    warnings.filterwarnings('always')
    with warnings.catch_warnings(record=True) as w_list:
        w = windows.tukey(M, alpha)
        has_nan = np.any(np.isnan(w))
        print(f"alpha={alpha:.2e}: NaN present = {has_nan}")
        if has_nan:
            print(f"  -> Result: {w}")