import numpy as np
from scipy.interpolate import PPoly

# Test 1: Constant zero polynomial as described in bug report
print("Test 1: Constant zero polynomial")
x = np.array([0.0, 1.0])
c = np.array([[0.0]])  # Constant zero polynomial
pp = PPoly(c, x)

print(f"Polynomial coefficients: {c}")
print(f"Breakpoints: {x}")
roots = pp.roots()
print(f"roots() returns: {roots}")
print(f"Documentation says: 'If sections are identically zero, root list contains start point followed by nan'")
print(f"Start point of interval: {x[0]}")
print(f"Does this match documentation? Start={roots[0]}, followed by nan={np.isnan(roots[1])}")
print()

# Test 2: Let's also test with solve() directly
print("Test 2: Using solve() directly")
solve_result = pp.solve(0.0)
print(f"solve(0.0) returns: {solve_result}")
print()

# Test 3: A different constant zero polynomial
print("Test 3: Constant zero on different interval")
x2 = np.array([5.0, 10.0])
c2 = np.array([[0.0]])
pp2 = PPoly(c2, x2)
roots2 = pp2.roots()
print(f"Interval: [{x2[0]}, {x2[1]}]")
print(f"roots() returns: {roots2}")
print(f"Expected per docs: start point ({x2[0]}) followed by nan")
print()

# Test 4: Multiple segments with one zero segment
print("Test 4: Piecewise with one zero segment")
x3 = np.array([0.0, 1.0, 2.0])
c3 = np.array([[1.0, 0.0], [0.0, 0.0]])  # First segment: y=1, Second segment: y=0
pp3 = PPoly(c3, x3)
roots3 = pp3.roots()
print(f"Segments: [0,1] has y=1, [1,2] has y=0")
print(f"roots() returns: {roots3}")
print(f"Expected: start of zero segment (1.0) followed by nan")