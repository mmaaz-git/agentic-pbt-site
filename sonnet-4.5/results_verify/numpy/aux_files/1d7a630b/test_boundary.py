import numpy as np
import numpy.lib.scimath as scimath

# Find the boundary where overflow to inf+nanj starts
test_values = [
    -1e-150, -1e-155, -1e-160, -1e-165, -1e-170,
    -1e-175, -1e-180, -1e-185, -1e-190, -1e-195, -1e-200
]

print("Finding overflow boundary for scimath.power(x, -2):\n")
print(f"{'x value':15} | {'Result':25} | Has NaN")
print("-" * 55)

for x in test_values:
    result = scimath.power(x, -2)
    has_nan = np.isnan(result)
    # Format result to show if it's inf or a regular number
    if np.isinf(np.real(result)):
        result_str = str(result)
    else:
        result_str = f"{np.real(result):.2e}+{np.imag(result):.2e}j"
    print(f"{x:15.2e} | {result_str:25} | {has_nan}")

# Also test what happens with direct complex arithmetic
print("\nDirect complex arithmetic for comparison:")
x = -1e-200 + 0j
print(f"x = {x}")
print(f"x ** 2 = {x ** 2}")
print(f"1 / (x ** 2) = {1 / (x ** 2)}")
print(f"x ** -2 = {x ** -2}")