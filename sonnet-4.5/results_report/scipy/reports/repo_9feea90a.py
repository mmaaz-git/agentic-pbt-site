from scipy import special
import numpy as np

x = 0.5
lmbda = 5e-324

y = special.boxcox(x, lmbda)
print(f"boxcox(0.5, 5e-324) = {y}")
print(f"Expected log(0.5) = {np.log(0.5)}")

result = special.inv_boxcox(y, lmbda)
print(f"inv_boxcox({y}, 5e-324) = {result}")
print(f"Expected: 0.5")
print(f"Actual: {result}")

print(f"\nWith exact zero:")
y_zero = special.boxcox(x, 0.0)
result_zero = special.inv_boxcox(y_zero, 0.0)
print(f"inv_boxcox(boxcox(0.5, 0.0), 0.0) = {result_zero}")