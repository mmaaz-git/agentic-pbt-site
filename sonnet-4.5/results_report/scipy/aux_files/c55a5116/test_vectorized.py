import numpy as np
from scipy.integrate import tanhsinh

# Test 1: Constant function that returns scalar
print("Test 1: Constant function returning scalar")
def f1(x):
    return 1.0

try:
    result = tanhsinh(f1, 0.0, 1.0)
    print(f"Success: {result.integral}")
except IndexError as e:
    print(f"Failed with IndexError: {e}")

# Test 2: Constant function that returns array properly
print("\nTest 2: Constant function returning array with same shape")
def f2(x):
    return np.ones_like(x)

try:
    result = tanhsinh(f2, 0.0, 1.0)
    print(f"Success: {result.integral}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: Debugging - see what shape x has
print("\nTest 3: Examining input shape")
def f3(x):
    print(f"Input x shape: {np.asarray(x).shape}")
    print(f"Input x type: {type(x)}")
    return np.ones_like(x)

try:
    result = tanhsinh(f3, 0.0, 1.0)
    print(f"Success: {result.integral}")
except Exception as e:
    print(f"Failed: {e}")