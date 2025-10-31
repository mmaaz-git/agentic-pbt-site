import numpy as np
import numpy.ma as ma

# Reproducing the bug
x = ma.array([np.inf], mask=[False])
y = ma.array([0.], mask=[True])

print(f'allclose(x, y) = {ma.allclose(x, y)}')
print(f'allclose(y, x) = {ma.allclose(y, x)}')

# Testing symmetry property
if ma.allclose(x, y) != ma.allclose(y, x):
    print("\nBUG CONFIRMED: allclose is not symmetric!")
    print(f"  allclose(x, y) = {ma.allclose(x, y)}")
    print(f"  allclose(y, x) = {ma.allclose(y, x)}")
else:
    print("\nNo asymmetry detected")