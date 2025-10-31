import numpy as np
from scipy import integrate

# Main example from bug report
x = np.array([1.0, 1.0, 2.0])
y = np.array([1.0, 1.0, 1.0])

result = integrate.simpson(y, x=x)
expected = 1.0

print("Main example:")
print(f"x: {x}")
print(f"y: {y}")
print(f"simpson result: {result}")
print(f"Expected: {expected}")
print(f"trapezoid result: {integrate.trapezoid(y, x=x)}")
print(f"Error: {result - expected}")
print()