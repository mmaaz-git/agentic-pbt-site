import numpy as np
from io import BytesIO
import scipy.io.matlab as matlab

print("Testing different empty array shapes with savemat/loadmat:")
print("=" * 70)

# Test various empty array shapes
test_arrays = [
    ("1D empty (0,)", np.array([])),
    ("2D row empty (1, 0)", np.zeros((1, 0))),
    ("2D column empty (0, 1)", np.zeros((0, 1))),
    ("2D empty (0, 0)", np.zeros((0, 0))),
    ("2D row non-empty (1, 3)", np.array([[1, 2, 3]])),
    ("2D column non-empty (3, 1)", np.array([[1], [2], [3]])),
]

for oned_as in ['row', 'column']:
    print(f"\nUsing oned_as='{oned_as}':")
    print("-" * 40)

    for desc, arr in test_arrays:
        f = BytesIO()
        matlab.savemat(f, {'arr': arr}, oned_as=oned_as)
        f.seek(0)
        loaded = matlab.loadmat(f)['arr']

        print(f"  {desc:25} -> {str(loaded.shape):10}")

        # For 1D arrays, check if oned_as has the expected effect
        if arr.ndim == 1:
            if arr.size == 0:
                if oned_as == 'row':
                    expected = (1, 0)
                else:
                    expected = (0, 1)
                is_correct = loaded.shape == expected
                print(f"    Expected for 1D empty: {expected}, Correct: {is_correct}")
            else:
                if oned_as == 'row':
                    expected = (1, arr.size)
                else:
                    expected = (arr.size, 1)
                is_correct = loaded.shape == expected
                print(f"    Expected for 1D non-empty: {expected}, Correct: {is_correct}")