import numpy as np

print("Checking the sum of flattop coefficients:")
print("=" * 50)

coeffs = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
print(f"Coefficients: {coeffs}")
print(f"Sum of coefficients: {sum(coeffs):.15f}")
print(f"Sum - 1.0 = {sum(coeffs) - 1.0:.15e}")

print("\nChecking the proposed fix coefficients:")
print("=" * 50)

fixed_coeffs = [0.21557894935326313, 0.41663157875010526,
                0.27726315716821054, 0.08357894674926315,
                0.006947367979157896]
print(f"Fixed coefficients: {fixed_coeffs}")
print(f"Sum of fixed coefficients: {sum(fixed_coeffs):.15f}")
print(f"Sum - 1.0 = {sum(fixed_coeffs) - 1.0:.15e}")

print("\nLet's examine the actual implementation:")
print("=" * 50)

# Looking at the actual scipy implementation
import scipy.signal.windows as windows
import inspect

# Get the source code
source = inspect.getsource(windows.flattop)
print("First 50 lines of flattop source:")
print(source[:2000] if len(source) > 2000 else source)