import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
from scipy.integrate import cumulative_simpson

print("Test Case 1: [0.0, 0.0, 1.0]")
y = np.array([0.0, 0.0, 1.0])
result = cumulative_simpson(y, initial=0)
print(f"Input: {y}")
print(f"Result: {result}")
print(f"Issue: result[1] = {result[1]} is negative for non-negative function")

print("\nTest Case 2: Linear function y=x")
y2 = np.array([0.0, 0.5, 1.0])
result2 = cumulative_simpson(y2, initial=0)
print(f"Input (linear y=x): {y2}")
print(f"Result: {result2}")
print(f"Expected: [0.0, 0.125, 0.5]")
print(f"Issue: result[1] = {result2[1]} should be 0.125")

print("\nTest Case 3: [1.0, 0.0, 0.0]")
y3 = np.array([1.0, 0.0, 0.0])
result3 = cumulative_simpson(y3, initial=0)
print(f"Input: {y3}")
print(f"Result: {result3}")
print(f"Issue: Non-monotonic - result[1] = {result3[1]} > result[2] = {result3[2]}")