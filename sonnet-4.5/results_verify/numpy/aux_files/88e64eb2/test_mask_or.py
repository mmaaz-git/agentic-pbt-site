import numpy as np
import numpy.ma as ma

# Test mask_or behavior
tests = [
    ("Both [True]", np.array([True]), np.array([True])),
    ("Both [False]", np.array([False]), np.array([False])),
    ("[True] and [False]", np.array([True]), np.array([False])),
    ("[False] and [True]", np.array([False]), np.array([True])),
]

print("Testing ma.mask_or:")
print("-" * 60)
for name, m1, m2 in tests:
    result = ma.mask_or(m1, m2)
    result_shape = result.shape if hasattr(result, 'shape') else 'scalar'
    print(f"{name:20} | Result shape: {str(result_shape):10} | Result: {result}")