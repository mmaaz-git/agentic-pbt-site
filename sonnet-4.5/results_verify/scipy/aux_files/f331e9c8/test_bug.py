import numpy as np
from scipy import integrate
import traceback

print("Testing invalid axis parameter on 2D array [[1, 2, 3], [4, 5, 6]]")
print("Array shape: (2, 3), ndim=2")
print("Testing with axis=2 (out of bounds)")
print("=" * 60)

y = np.array([[1, 2, 3],
              [4, 5, 6]])

print("\n1. Testing trapezoid with invalid axis=2:")
try:
    result = integrate.trapezoid(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("\n2. Testing simpson with invalid axis=2:")
try:
    result = integrate.simpson(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("\n3. Testing cumulative_trapezoid with invalid axis=2:")
try:
    result = integrate.cumulative_trapezoid(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("\n4. Testing cumulative_simpson with invalid axis=2 (for comparison):")
try:
    result = integrate.cumulative_simpson(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("\n5. Testing numpy.sum with invalid axis=2 (for comparison):")
try:
    result = np.sum(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

print("\n" + "=" * 60)
print("\nTesting with negative out-of-bounds axis=-3:")
print("\n6. Testing trapezoid with invalid axis=-3:")
try:
    result = integrate.trapezoid(y, axis=-3)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")