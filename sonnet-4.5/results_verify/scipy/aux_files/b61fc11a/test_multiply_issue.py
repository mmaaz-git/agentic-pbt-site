import numpy as np

# Test the problematic operation: res + (res_j * 1j)
# where res contains real parts and res_j contains imaginary parts

# Simulate what happens in the scipy code
res = np.array([[0.0]])  # Real part
res_j = np.array([[np.inf]])  # Imaginary part

print(f"res (real part): {res}")
print(f"res_j (imag part): {res_j}")

# The problematic operation
result = res + (res_j * 1j)
print(f"\nResult of res + (res_j * 1j): {result}")
print(f"Result real part: {result[0,0].real}")
print(f"Result imag part: {result[0,0].imag}")

# Check if the warning occurs
import warnings
warnings.filterwarnings('error')
try:
    result2 = res + (res_j * 1j)
    print("No warning raised")
except RuntimeWarning as e:
    print(f"\nRuntimeWarning raised: {e}")

# Alternative approach that might work better
print("\n--- Alternative approach ---")
result_alt = np.empty(res.shape, dtype=np.complex128)
result_alt.real = res
result_alt.imag = res_j
print(f"Alternative result: {result_alt}")
print(f"Alternative real part: {result_alt[0,0].real}")
print(f"Alternative imag part: {result_alt[0,0].imag}")