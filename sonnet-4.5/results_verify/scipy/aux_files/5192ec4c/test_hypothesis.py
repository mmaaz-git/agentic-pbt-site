import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import is_valid_dm


@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=100)
def test_is_valid_dm_error_handling_without_name(n):
    mat = np.eye(n) * 5.0 + np.ones((n, n))

    try:
        is_valid_dm(mat, tol=0.1, throw=True, name=None)
        assert False, "Should raise ValueError for non-zero diagonal"
    except ValueError:
        pass
    except TypeError as e:
        print(f"Got TypeError instead of ValueError for n={n}: {e}")
        raise

if __name__ == "__main__":
    test_is_valid_dm_error_handling_without_name()