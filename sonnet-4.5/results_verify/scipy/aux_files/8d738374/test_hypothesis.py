import numpy as np
from io import BytesIO
from hypothesis import given, strategies as st, settings
import scipy.io.matlab as matlab

@given(oned_as=st.sampled_from(['row', 'column']))
@settings(max_examples=10)
def test_oned_as_consistency_empty_arrays(oned_as):
    arr = np.array([])

    f = BytesIO()
    matlab.savemat(f, {'arr': arr}, oned_as=oned_as)
    f.seek(0)
    loaded = matlab.loadmat(f)
    result = loaded['arr']

    if oned_as == 'row':
        expected_shape = (1, 0)
    else:
        expected_shape = (0, 1)

    print(f"Testing oned_as='{oned_as}': expected shape {expected_shape}, got {result.shape}")

    assert result.shape == expected_shape, (
        f"oned_as='{oned_as}' should produce shape {expected_shape}, "
        f"but got {result.shape}"
    )

if __name__ == "__main__":
    print("Running hypothesis test for oned_as parameter with empty arrays:")
    print("-" * 60)
    try:
        test_oned_as_consistency_empty_arrays()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")