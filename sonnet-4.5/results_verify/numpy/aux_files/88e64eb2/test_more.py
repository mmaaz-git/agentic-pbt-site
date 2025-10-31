import numpy as np
import numpy.ma as ma

# Test various inputs
tests = [
    ("Single True", np.array([True])),
    ("Single False", np.array([False])),
    ("Two Trues", np.array([True, True])),
    ("Two Falses", np.array([False, False])),
    ("Mixed", np.array([True, False])),
    ("Empty array", np.array([], dtype=bool)),
    ("2D single False", np.array([[False]])),
    ("2D all False", np.array([[False, False], [False, False]])),
    ("Scalar True", True),
    ("Scalar False", False),
]

print("Testing with shrink=True (default):")
print("-" * 50)
for name, input_val in tests:
    result = ma.make_mask(input_val, shrink=True)
    input_shape = input_val.shape if hasattr(input_val, 'shape') else 'scalar'
    result_shape = result.shape if hasattr(result, 'shape') else 'scalar'
    print(f"{name:20} | Input shape: {str(input_shape):10} | Result shape: {str(result_shape):10} | Result: {result}")

print("\nTesting with shrink=False:")
print("-" * 50)
for name, input_val in tests:
    result = ma.make_mask(input_val, shrink=False)
    input_shape = input_val.shape if hasattr(input_val, 'shape') else 'scalar'
    result_shape = result.shape if hasattr(result, 'shape') else 'scalar'
    print(f"{name:20} | Input shape: {str(input_shape):10} | Result shape: {str(result_shape):10} | Result: {result}")