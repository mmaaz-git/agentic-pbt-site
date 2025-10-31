import scipy.special
import numpy as np

# Test case demonstrating the bug
x = 1.0
lmbda = 1e-264

# Apply boxcox1p transformation
y = scipy.special.boxcox1p(x, lmbda)
print(f"boxcox1p({x}, {lmbda}) = {y}")

# Apply inverse transformation
result = scipy.special.inv_boxcox1p(y, lmbda)
print(f"inv_boxcox1p({y}, {lmbda}) = {result}")
print(f"Expected: {x}")
print(f"Error: {abs(result - x)}")

# Additional test with exactly lambda = 0 to show it works correctly
print("\n--- Testing with lambda = 0 (should work correctly) ---")
y_zero = scipy.special.boxcox1p(x, 0.0)
result_zero = scipy.special.inv_boxcox1p(y_zero, 0.0)
print(f"boxcox1p({x}, 0.0) = {y_zero}")
print(f"inv_boxcox1p({y_zero}, 0.0) = {result_zero}")
print(f"Expected: {x}")
print(f"Error: {abs(result_zero - x)}")

# Test with various small lambda values to find threshold
print("\n--- Testing various small lambda values ---")
for exp in [-100, -200, -250, -260, -264, -300]:
    test_lmbda = 10.0 ** exp
    y_test = scipy.special.boxcox1p(x, test_lmbda)
    result_test = scipy.special.inv_boxcox1p(y_test, test_lmbda)
    error = abs(result_test - x)
    status = "OK" if error < 1e-10 else "FAIL"
    print(f"lambda=1e{exp}: error={error:.15e} [{status}]")