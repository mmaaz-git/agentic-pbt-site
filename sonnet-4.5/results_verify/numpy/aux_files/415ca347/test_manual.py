import numpy as np

a = 3_036_988_439
b = 3_037_012_561

print(f"Testing np.lcm({a}, {b})")
lcm_val = np.lcm(a, b)
print(f"np.lcm({a}, {b}) = {lcm_val}")

# Check what Python's math.lcm gives us
import math
python_lcm = math.lcm(a, b)
print(f"Python's math.lcm({a}, {b}) = {python_lcm}")

# Check GCD * LCM = a * b relationship
gcd_val = np.gcd(a, b)
print(f"np.gcd({a}, {b}) = {gcd_val}")
print(f"Product (GCD * LCM) = {gcd_val * lcm_val}")
print(f"Expected (a * b) = {a * b}")

# Check if LCM is positive
assert lcm_val > 0, f"LCM of positive integers should be positive, got {lcm_val}"