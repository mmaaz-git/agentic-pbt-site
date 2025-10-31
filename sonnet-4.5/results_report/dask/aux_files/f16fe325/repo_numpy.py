import numpy as np
import numpy.ma as ma

# Create test arrays
x = ma.array([np.inf], mask=[False])
y = ma.array([0.], mask=[True])

print("Testing numpy.ma.allclose asymmetry with masked values and infinity:")
print(f"x = ma.array([np.inf], mask=[False])")
print(f"y = ma.array([0.], mask=[True])")
print()
print(f"ma.allclose(x, y) = {ma.allclose(x, y)}")
print(f"ma.allclose(y, x) = {ma.allclose(y, x)}")
print()
print("Expected: Both should return the same value (symmetric)")
print("Actual: They return different values (asymmetric)")