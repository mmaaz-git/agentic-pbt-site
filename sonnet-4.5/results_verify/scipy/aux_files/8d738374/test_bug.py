import numpy as np
from io import BytesIO
import scipy.io.matlab as matlab

print("Testing scipy.io.matlab.savemat oned_as parameter with empty arrays")
print("=" * 70)

# Test with empty array
arr_empty = np.array([])
arr_nonempty = np.array([1, 2, 3])

print(f"Original empty array shape: {arr_empty.shape}")
print(f"Original non-empty array shape: {arr_nonempty.shape}")

for oned_as in ['row', 'column']:
    print(f"\nTesting with oned_as='{oned_as}':")

    # Test empty array
    f_empty = BytesIO()
    matlab.savemat(f_empty, {'arr': arr_empty}, oned_as=oned_as)
    f_empty.seek(0)
    loaded_empty = matlab.loadmat(f_empty)['arr']

    # Test non-empty array
    f_nonempty = BytesIO()
    matlab.savemat(f_nonempty, {'arr': arr_nonempty}, oned_as=oned_as)
    f_nonempty.seek(0)
    loaded_nonempty = matlab.loadmat(f_nonempty)['arr']

    print(f"  Non-empty [1,2,3]: (3,) -> {loaded_nonempty.shape}")
    print(f"  Empty []:          (0,) -> {loaded_empty.shape}")

    if oned_as == 'row':
        expected_empty_shape = (1, 0)
        print(f"  Expected empty: {expected_empty_shape}, Got: {loaded_empty.shape}")
        print(f"  Matches expectation: {loaded_empty.shape == expected_empty_shape}")
    else:  # column
        expected_empty_shape = (0, 1)
        print(f"  Expected empty: {expected_empty_shape}, Got: {loaded_empty.shape}")
        print(f"  Matches expectation: {loaded_empty.shape == expected_empty_shape}")