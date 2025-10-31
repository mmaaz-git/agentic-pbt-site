import numpy as np
from hypothesis import given, strategies as st
from numpy.matrixlib import bmat, matrix
import hypothesis.extra.numpy as npst


@given(
    st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=10),
    npst.arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 3), st.integers(1, 3)))
)
def test_bmat_gdict_without_ldict(varname, arr):
    """Test that bmat works with gdict parameter but without ldict.

    Since ldict has a default value of None in the function signature,
    it should be optional when using gdict.
    """
    m = matrix(arr)
    gdict = {varname: m}
    # This should work but crashes with TypeError
    result = bmat(varname, gdict=gdict)
    np.testing.assert_array_equal(result, m)

if __name__ == "__main__":
    # Run the test
    test_bmat_gdict_without_ldict()