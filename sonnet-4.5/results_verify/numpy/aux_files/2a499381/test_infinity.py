import numpy as np
import numpy.ma as ma

# Test how masked and unmasked allclose handle infinities

print("Testing numpy.allclose (non-masked) with infinities:")
x1 = np.array([np.inf])
y1 = np.array([np.inf])
print(f"  allclose([inf], [inf]) = {np.allclose(x1, y1)}")

x2 = np.array([np.inf])
y2 = np.array([1.0])
print(f"  allclose([inf], [1.0]) = {np.allclose(x2, y2)}")

print("\nTesting numpy.ma.allclose with infinities and no masks:")
x3 = ma.array([np.inf])
y3 = ma.array([np.inf])
print(f"  ma.allclose([inf], [inf]) = {ma.allclose(x3, y3)}")

x4 = ma.array([np.inf])
y4 = ma.array([1.0])
print(f"  ma.allclose([inf], [1.0]) = {ma.allclose(x4, y4)}")

print("\nTesting with masked infinities:")
x5 = ma.array([np.inf], mask=[True])
y5 = ma.array([np.inf], mask=[True])
print(f"  ma.allclose([inf (masked)], [inf (masked)]) = {ma.allclose(x5, y5)}")

x6 = ma.array([np.inf], mask=[True])
y6 = ma.array([1.0], mask=[False])
print(f"  ma.allclose([inf (masked)], [1.0 (unmasked)]) = {ma.allclose(x6, y6)}")
print(f"  ma.allclose([1.0 (unmasked)], [inf (masked)]) = {ma.allclose(y6, x6)}")

x7 = ma.array([np.inf], mask=[False])
y7 = ma.array([1.0], mask=[True])
print(f"  ma.allclose([inf (unmasked)], [1.0 (masked)]) = {ma.allclose(x7, y7)}")
print(f"  ma.allclose([1.0 (masked)], [inf (unmasked)]) = {ma.allclose(y7, x7)}")