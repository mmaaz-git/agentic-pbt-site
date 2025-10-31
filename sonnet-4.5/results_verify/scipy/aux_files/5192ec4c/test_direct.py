import numpy as np
from scipy.spatial.distance import is_valid_dm

mat = np.array([[5.0, 1.0], [1.0, 5.0]])
print("Testing with name=None:")
try:
    is_valid_dm(mat, tol=0.1, throw=True, name=None)
except TypeError as e:
    print(f"Got TypeError: {e}")
except ValueError as e:
    print(f"Got ValueError: {e}")

print("\nTesting with name='test':")
try:
    is_valid_dm(mat, tol=0.1, throw=True, name='test')
except TypeError as e:
    print(f"Got TypeError: {e}")
except ValueError as e:
    print(f"Got ValueError: {e}")