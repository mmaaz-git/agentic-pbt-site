import scipy.special as sp
import numpy as np

print("=" * 60)
print("Reproducing the bug with extremely small lambda")
print("=" * 60)

x = 1.0
lmbda = 1e-300

y = sp.boxcox1p(x, lmbda)
result = sp.inv_boxcox1p(y, lmbda)

print(f"boxcox1p({x}, {lmbda}) = {y}")
print(f"inv_boxcox1p({y}, {lmbda}) = {result}")
print(f"Expected: {x}")
print(f"Actual: {result}")
print(f"Error: {abs(result - x)}")
print()

print("=" * 60)
print("Testing with different lambda values")
print("=" * 60)

lambdas = [0.0, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300, 1e-308]

for lmbda in lambdas:
    y = sp.boxcox1p(x, lmbda)
    result = sp.inv_boxcox1p(y, lmbda)
    error = abs(result - x)
    status = "✓" if error < 1e-10 else "✗"
    print(f"lambda={lmbda:e}: error={error:e} {status}")

print()
print("=" * 60)
print("Testing the special case lambda=0")
print("=" * 60)

# When lambda=0, boxcox1p should give log(1+x)
lmbda = 0.0
y_expected = np.log(1 + x)
y = sp.boxcox1p(x, lmbda)
result = sp.inv_boxcox1p(y, lmbda)

print(f"lambda=0 case:")
print(f"  boxcox1p({x}, {lmbda}) = {y} (expected log(1+{x}) = {y_expected})")
print(f"  inv_boxcox1p({y}, {lmbda}) = {result} (expected {x})")
print(f"  Round-trip error: {abs(result - x)}")
print()

print("=" * 60)
print("Testing when lambda is close to zero but not exactly zero")
print("=" * 60)

# Let's find the threshold where the bug appears
test_lambdas = [10**(-i) for i in range(50, 310, 10)]
for lmbda in test_lambdas:
    y = sp.boxcox1p(x, lmbda)
    result = sp.inv_boxcox1p(y, lmbda)
    error = abs(result - x)
    if error > 1e-10:
        print(f"Bug appears at lambda <= {lmbda:e}")