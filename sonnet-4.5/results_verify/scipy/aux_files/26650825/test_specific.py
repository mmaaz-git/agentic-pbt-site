from scipy.special import boxcox1p, inv_boxcox1p
import math

y = 1.0
lmbda = 5.808166112732823e-234

x = inv_boxcox1p(y, lmbda)
y_recovered = boxcox1p(x, lmbda)

print(f"Input: y={y}, lambda={lmbda}")
print(f"inv_boxcox1p(y, lambda) = {x}")
print(f"boxcox1p(x, lambda) = {y_recovered}")
print(f"Expected: {y}")
print(f"Actual: {y_recovered}")
print(f"Error: {abs(y_recovered - y)}")
print(f"Is y_recovered equal to log(2)? {y_recovered == math.log(2)}")

# Test with another small lambda value
print("\nTesting with another very small lambda:")
lmbda2 = 1e-200
x2 = inv_boxcox1p(y, lmbda2)
y_recovered2 = boxcox1p(x2, lmbda2)
print(f"lambda={lmbda2}, x={x2}, y_recovered={y_recovered2}, error={abs(y_recovered2 - y)}")

# Test edge cases to understand the threshold
print("\nTesting to find the threshold:")
test_lambdas = [1e-100, 1e-150, 1e-157, 1e-158, 1e-200, 1e-250, 1e-300]
for test_lambda in test_lambdas:
    x_test = inv_boxcox1p(1.0, test_lambda)
    y_test = boxcox1p(x_test, test_lambda)
    print(f"lambda={test_lambda:e}, inv_boxcox1p(1.0)={x_test}, boxcox1p(x)={y_test:.6f}, error={abs(y_test - 1.0):.6f}")