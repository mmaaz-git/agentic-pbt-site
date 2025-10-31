import numpy as np
from scipy import integrate

print("Testing additional failing cases:")
print("=" * 50)

# Test case 1: x=[0.0, 0.0, 1.0], y=[2.0, 2.0, 2.0]
x1 = np.array([0.0, 0.0, 1.0])
y1 = np.array([2.0, 2.0, 2.0])
result1 = integrate.simpson(y1, x=x1)
expected1 = 2.0 * (1.0 - 0.0)  # c*(b-a) = 2.0 * 1.0
print(f"Case 1: x={x1}, y={y1}")
print(f"simpson result: {result1}")
print(f"Expected: {expected1}")
print(f"trapezoid result: {integrate.trapezoid(y1, x=x1)}")
print(f"Error: {result1 - expected1}")
print()

# Test case 2: x=[0.0, 0.0, 0.0, 1.0], y=[5.0, 5.0, 5.0, 5.0]
x2 = np.array([0.0, 0.0, 0.0, 1.0])
y2 = np.array([5.0, 5.0, 5.0, 5.0])
result2 = integrate.simpson(y2, x=x2)
expected2 = 5.0 * (1.0 - 0.0)  # c*(b-a) = 5.0 * 1.0
print(f"Case 2: x={x2}, y={y2}")
print(f"simpson result: {result2}")
print(f"Expected: {expected2}")
print(f"trapezoid result: {integrate.trapezoid(y2, x=x2)}")
print(f"Error: {result2 - expected2}")
print()

# Test case 3: x=[1.0, 2.0, 2.0], y=[3.0, 3.0, 3.0]
x3 = np.array([1.0, 2.0, 2.0])
y3 = np.array([3.0, 3.0, 3.0])
result3 = integrate.simpson(y3, x=x3)
expected3 = 3.0 * (2.0 - 1.0)  # c*(b-a) = 3.0 * 1.0
print(f"Case 3: x={x3}, y={y3}")
print(f"simpson result: {result3}")
print(f"Expected: {expected3}")
print(f"trapezoid result: {integrate.trapezoid(y3, x=x3)}")
print(f"Error: {result3 - expected3}")
print()

# Test with no duplicates (should work correctly)
x4 = np.array([0.0, 1.0, 2.0])
y4 = np.array([3.0, 3.0, 3.0])
result4 = integrate.simpson(y4, x=x4)
expected4 = 3.0 * (2.0 - 0.0)  # c*(b-a) = 3.0 * 2.0
print(f"Control case (no duplicates): x={x4}, y={y4}")
print(f"simpson result: {result4}")
print(f"Expected: {expected4}")
print(f"trapezoid result: {integrate.trapezoid(y4, x=x4)}")
print(f"Error: {result4 - expected4}")
print()