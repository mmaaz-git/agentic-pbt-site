import numpy as np
from hypothesis import given, strategies as st
from numpy.matrixlib import bmat, matrix
import hypothesis.extra.numpy as npst


@given(
    st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=10),
    npst.arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 3), st.integers(1, 3)))
)
def test_bmat_gdict_without_ldict(varname, arr):
    m = matrix(arr)
    gdict = {varname: m}
    result = bmat(varname, gdict=gdict)
    np.testing.assert_array_equal(result, m)

if __name__ == "__main__":
    test_bmat_gdict_without_ldict()