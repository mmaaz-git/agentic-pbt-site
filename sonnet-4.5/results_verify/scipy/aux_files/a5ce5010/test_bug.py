import numpy as np
from scipy.interpolate import PPoly

print("Testing PPoly antiderivative bug:")
print("-" * 50)

c = np.array([[1.0, 2.0]])
x = np.array([0.0, 1.0, 2.0])
p = PPoly(c, x)

p_deriv = p.derivative()
p_deriv_anti = p_deriv.antiderivative()

print(f"Original piecewise constant function p:")
print(f"  p(0.5) = {p(0.5)}")
print(f"  p(1.5) = {p(1.5)}")
print()

print(f"After derivative().antiderivative():")
print(f"  p'_anti(0.5) = {p_deriv_anti(0.5)}")
print(f"  p'_anti(1.5) = {p_deriv_anti(1.5)}")
print()

diff_at_05 = p(0.5) - p_deriv_anti(0.5)
diff_at_15 = p(1.5) - p_deriv_anti(1.5)
print(f"Difference between original and reconstructed:")
print(f"  At x=0.5: {diff_at_05}")
print(f"  At x=1.5: {diff_at_15}")
print()

if abs(diff_at_05 - diff_at_15) > 1e-10:
    print(f"BUG CONFIRMED: Difference changes from {diff_at_05} to {diff_at_15}")
    print("Expected: The difference should be constant!")
else:
    print("No bug: Difference is constant")