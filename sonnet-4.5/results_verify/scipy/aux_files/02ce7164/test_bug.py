import numpy as np
import scipy.differentiate

def f(x):
    return x**2

x = 1.0

print("Testing initial_step=0:")
result = scipy.differentiate.derivative(f, x, initial_step=0)
print(f"  success: {result.success}")
print(f"  df: {result.df}")
print(f"  status: {result.status}")

print("\nTesting initial_step=-0.5:")
result2 = scipy.differentiate.derivative(f, x, initial_step=-0.5)
print(f"  success: {result2.success}")
print(f"  df: {result2.df}")
print(f"  status: {result2.status}")

print("\nTesting initial_step=0.5 (valid):")
result3 = scipy.differentiate.derivative(f, x, initial_step=0.5)
print(f"  success: {result3.success}")
print(f"  df: {result3.df}")
print(f"  status: {result3.status}")