import numpy as np
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst


@given(
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
)
def test_bmat_gdict_without_ldict_crashes(arr1, arr2):
    m1 = np.matrix(arr1)
    m2 = np.matrix(arr2)
    global_vars = {'X': m1, 'Y': m2}
    result = np.bmat('X,Y', gdict=global_vars)
    expected = np.bmat([[m1, m2]])
    assert np.array_equal(result, expected)

# Run the test
if __name__ == "__main__":
    test_bmat_gdict_without_ldict_crashes()