import numpy as np
from scipy import special

lmbda = -10
x = 26

y = special.boxcox(x, lmbda)
x_recovered = special.inv_boxcox(y, lmbda)

print(f"Original x: {x}")
print(f"Transformed y: {y}")
print(f"Recovered x: {x_recovered}")
print(f"Relative error: {abs(x - x_recovered) / x:.6f}")
print(f"Absolute error: {abs(x - x_recovered):.6f}")

print("\nAdditional failing cases:")

# Test case 2
lmbda2 = -10
x2 = 50
y2 = special.boxcox(x2, lmbda2)
x2_recovered = special.inv_boxcox(y2, lmbda2)
print(f"\nlmbda={lmbda2}, x={x2}:")
print(f"  Recovered x: {x2_recovered}")
print(f"  Is inf: {np.isinf(x2_recovered)}")

# Test case 3
lmbda3 = -10
x3 = 100
y3 = special.boxcox(x3, lmbda3)
x3_recovered = special.inv_boxcox(y3, lmbda3)
print(f"\nlmbda={lmbda3}, x={x3}:")
print(f"  Recovered x: {x3_recovered}")
print(f"  Is inf: {np.isinf(x3_recovered)}")

# Test case 4
lmbda4 = -8
x4 = 100
y4 = special.boxcox(x4, lmbda4)
x4_recovered = special.inv_boxcox(y4, lmbda4)
print(f"\nlmbda={lmbda4}, x={x4}:")
print(f"  Recovered x: {x4_recovered}")
if not np.isinf(x4_recovered):
    print(f"  Relative error: {abs(x4 - x4_recovered) / x4:.6f}")