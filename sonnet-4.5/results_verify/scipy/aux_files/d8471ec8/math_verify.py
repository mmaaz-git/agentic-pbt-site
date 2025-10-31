#!/usr/bin/env python3
import numpy as np
from scipy.integrate import cumulative_simpson

# Let's manually verify the mathematics for y = [0, 0, 1]
# with equal spacing dx = 1

y = np.array([0.0, 0.0, 1.0])
x = np.array([0.0, 1.0, 2.0])

print("Manual calculation for y = [0, 0, 1] with x = [0, 1, 2]")
print("=" * 60)

# Cumulative integral should give us the integral from x[0] to x[i] for each i
# For non-negative functions, this should be monotonic increasing

# For Simpson's rule with 3 points (quadratic fit), the integral from a to c is:
# (c-a)/6 * [f(a) + 4*f((a+c)/2) + f(c)]
# But this is for the full interval

# Let's trace through what cumulative_simpson actually computes
result = cumulative_simpson(y, initial=0)
print(f"cumulative_simpson result: {result}")
print(f"This claims: integral from 0 to 0 = 0")
print(f"             integral from 0 to 1 = {result[1]}")
print(f"             integral from 0 to 2 = {result[2]}")

# Mathematical fact: for f(x) >= 0, the integral from a to b where b > a
# MUST be >= 0. The cumulative integral MUST be non-decreasing.

# Let's check what it means to have integral from 0 to 1 = -0.0833...
print(f"\nIssue: The integral from 0 to 1 is negative: {result[1]}")
print("But y >= 0 everywhere, so the integral CANNOT be negative!")

# Let's also manually compute using basic trapezoid rule to verify
# Trapezoid from 0 to 1: area = 0.5 * (y[0] + y[1]) * (x[1] - x[0]) = 0.5 * (0 + 0) * 1 = 0
# Trapezoid from 1 to 2: area = 0.5 * (y[1] + y[2]) * (x[2] - x[1]) = 0.5 * (0 + 1) * 1 = 0.5
print("\nManual trapezoid calculation:")
print(f"Integral 0 to 1: 0.5 * (0 + 0) * 1 = 0")
print(f"Integral 0 to 2: 0 + 0.5 * (0 + 1) * 1 = 0.5")

# The cumulative integral of a non-negative function MUST be non-decreasing
# This is a fundamental property from calculus
print("\n" + "=" * 60)
print("FUNDAMENTAL THEOREM:")
print("If f(x) >= 0 for all x in [a,b], then")
print("F(x) = integral from a to x of f(t) dt")
print("is a non-decreasing function.")
print("This means F(x2) >= F(x1) whenever x2 > x1.")
print("=" * 60)