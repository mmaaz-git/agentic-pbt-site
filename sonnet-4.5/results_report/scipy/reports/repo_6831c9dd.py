import numpy as np
import scipy.integrate as integrate

# Test case from the bug report
x = np.array([0.0, 1.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 1.0, 2.0])

result = integrate.simpson(y, x=x)

print(f"simpson result: {result}")
print(f"Expected: 2.0")
print()
print(f"Explanation: When integrating y=x from 0 to 2, the mathematical result")
print(f"should be ∫₀² x dx = x²/2 |₀² = 4/2 - 0/2 = 2.0")
print()
print(f"The duplicate x value at indices 1 and 2 (both equal to 1.0)")
print(f"represents a zero-width segment that should contribute 0 to the integral.")
print(f"However, simpson produces the result {result} instead of the correct value 2.0.")