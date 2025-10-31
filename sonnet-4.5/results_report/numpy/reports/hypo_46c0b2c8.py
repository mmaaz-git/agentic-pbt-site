import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
       npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
       st.data())
def test_intersect1d_masked_handling(ar1, ar2, data):
    assume(ar1.size > 1 and ar2.size > 1)

    mask1 = data.draw(npst.arrays(dtype=np.bool_, shape=ar1.shape))
    mask2 = data.draw(npst.arrays(dtype=np.bool_, shape=ar2.shape))

    assume(np.sum(mask1) >= 1 and np.sum(mask2) >= 1)

    mar1 = ma.array(ar1, mask=mask1)
    mar2 = ma.array(ar2, mask=mask2)

    intersection = ma.intersect1d(mar1, mar2)

    masked_in_result = ma.getmaskarray(intersection)
    if masked_in_result is not ma.nomask:
        assert np.sum(masked_in_result) <= 1, f"Found {np.sum(masked_in_result)} masked values, expected at most 1"

if __name__ == "__main__":
    test_intersect1d_masked_handling()