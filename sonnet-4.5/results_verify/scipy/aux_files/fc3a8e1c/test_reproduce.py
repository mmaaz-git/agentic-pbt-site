import numpy as np
from scipy.differentiate import derivative

print("Testing derivative with initial_step=0.0")
result = derivative(np.sin, 1.0, initial_step=0.0)
print(f"success: {result.success}")
print(f"status: {result.status}")
print(f"df: {result.df}")
print(f"nfev: {result.nfev}")
print(f"nit: {result.nit}")

print("\nTesting derivative with initial_step=-1.0")
result = derivative(np.sin, 1.0, initial_step=-1.0)
print(f"success: {result.success}")
print(f"status: {result.status}")
print(f"df: {result.df}")
print(f"nfev: {result.nfev}")
print(f"nit: {result.nit}")

print("\nTesting derivative with initial_step=1.0 (positive, should work)")
result = derivative(np.sin, 1.0, initial_step=1.0)
print(f"success: {result.success}")
print(f"status: {result.status}")
print(f"df: {result.df}")
print(f"nfev: {result.nfev}")
print(f"nit: {result.nit}")