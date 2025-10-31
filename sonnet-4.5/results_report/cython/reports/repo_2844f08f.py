import numpy as np
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

# Create simple test arrays
left = np.array([True, False])
right = np.array([True, True])

# Test kleene_and with both masks as None
print("Testing kleene_and with both masks as None:")
try:
    result, mask = kleene_and(left, right, None, None)
    print(f"Result: {result}, Mask: {mask}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
except Exception as e:
    print(f"Error: {e}")

# Test kleene_or with both masks as None
print("\nTesting kleene_or with both masks as None:")
try:
    result, mask = kleene_or(left, right, None, None)
    print(f"Result: {result}, Mask: {mask}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
except Exception as e:
    print(f"Error: {e}")

# Test kleene_xor with both masks as None
print("\nTesting kleene_xor with both masks as None:")
try:
    result, mask = kleene_xor(left, right, None, None)
    print(f"Result: {result}, Mask: {mask}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
except Exception as e:
    print(f"Error: {e}")